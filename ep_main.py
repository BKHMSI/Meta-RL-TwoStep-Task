import os
import yaml
import pickle
import argparse
import datetime
import scipy.signal

import numpy as np
import torch as T
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm 
from collections import namedtuple

from models.a2c_dnd_lstm import A2C_DND_LSTM
from tasks.ep_two_step import EpTwoStepTask

Rollout = namedtuple('Rollout',
                        ('state', 'action', 'reward', 'timestep', 'done', 'policy', 'value'))

class Trainer: 
    def __init__(self, config):
        self.device = 'cpu'
        self.seed = config["seed"]

        T.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        T.random.manual_seed(config["seed"])

        self.env = EpTwoStepTask(config["task"])  
        self.agent = A2C_DND_LSTM(
            self.env.feat_size,
            config["agent"]["mem-units"], 
            self.env.num_actions,
            config["agent"]["dict-len"],
            config["agent"]["dict-kernel"]
        ).to(self.device)

        self.optim = T.optim.Adam(self.agent.parameters(), lr=config["agent"]["lr"])

        self.val_coeff = config["agent"]["value-loss-weight"]
        self.entropy_coeff = config["agent"]["entropy-weight"]
        self.max_grad_norm = config["agent"]["max-grad-norm"]
        self.switch_p = config["task"]["swtich-prob"]
        self.start_episode = 0

        self.writer = SummaryWriter(log_dir=os.path.join("logs_ep", config["run-title"]))
        self.save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}")

        if config["resume"]:
            print("> Loading Checkpoint")
            self.start_episode = config["start-episode"]
            self.agent.load_state_dict(T.load(self.save_path.format(epi=self.start_episode) + ".pt")["state_dict"])

    def run_episode(self, episode):
        done = False
        total_reward = 0
        p_action, p_reward, timestep = [0,0], 0, 0

        self.agent.reset_memory()

        state = self.env.reset()
        (h_t, c_t) = self.agent.get_init_states()

        buffer = []

        while not done:

            # switch reward contingencies at the beginning of each episode with probability p
            self.env.possible_switch(switch_p=self.switch_p)

            trial = self.env.get_trial()
            if trial == "cued":
                self.agent.turn_off_encoding()
                self.agent.turn_on_retrieval()
            else:
                self.agent.turn_on_encoding()
                self.agent.turn_off_retrieval() 

            cue = self.env.get_cue()
            cue = T.tensor(cue, device=self.device)

            # sample action using model
            x_t = (
                T.tensor([state], device=self.device).float(), 
                T.tensor([p_action], device=self.device).float(),  
                T.tensor([[p_reward]], device=self.device).float(),  
                T.tensor([[timestep]], device=self.device).float(),
            )

            action_dist, values, (h_t, c_t) = self.agent(x_t, cue, (h_t, c_t))
            
            action_cat = T.distributions.Categorical(action_dist.squeeze())
            action = action_cat.sample()
            action_onehot = np.eye(2)[action]

            # take action and observe result
            new_state, reward, done, timestep, context = self.env.step(int(action), cue.numpy())

            context = T.tensor(context, device=self.device)
            self.agent.save_memory(context, c_t)

            # ('state', 'action', 'reward', 'timestep', 'done', 'policy', 'value')
            buffer += [Rollout(
                state, 
                action_onehot,
                reward,
                timestep,
                done,
                action_dist,
                values
            )]

            state = new_state
            p_reward = reward
            p_action = action_onehot

            total_reward += reward

        # boostrap final observation 
        cue = self.env.get_cue()
        cue = T.tensor(cue, device=self.device)
        _, values, _ = self.agent((
            T.tensor([state], device=self.device).float(), 
            T.tensor([p_action], device=self.device).float(),  
            T.tensor([[p_reward]], device=self.device).float(),  
            T.tensor([[timestep]], device=self.device).float(),
        ), cue, (h_t, c_t))

        buffer += [Rollout(None, None, None, None, None, None, values)]

        return total_reward, buffer

    def a2c_loss(self, buffer, gamma, lambd=1.0):
        # bootstrap discounted returns with final value estimates
        _, _, _, _, _, _, last_value = buffer[-1]
        returns = last_value.data
        advantages = 0

        all_returns = T.zeros(len(buffer)-1, device=self.device)
        all_advantages = T.zeros(len(buffer)-1, device=self.device)
        # run Generalized Advantage Estimation, calculate returns, advantages
        for t in reversed(range(len(buffer) - 1)):
            # ('state', 'action', 'reward', 'timestep', 'done', 'policy', 'value')
            _, _, reward, _, done, _, value = buffer[t]

            _, _, _, _, _, _, next_value = buffer[t+1]

            mask = ~done

            returns = reward + returns * gamma * mask

            deltas = reward + next_value.data * gamma * mask - value.data
            advantages = advantages * gamma * lambd * mask + deltas

            all_returns[t] = returns 
            all_advantages[t] = advantages

        batch = Rollout(*zip(*buffer))

        policy = T.cat(batch.policy[:-1], dim=0).squeeze().to(self.device)
        action = T.tensor(batch.action[:-1], device=self.device)
        values = T.tensor(batch.value[:-1], device=self.device)
        
        logits = (policy * action).sum(1)
        policy_loss = -(T.log(logits) * all_advantages).mean()
        value_loss = 0.5 * (all_returns - values).pow(2).mean()
        entropy_reg = -(policy * T.log(policy)).mean()

        loss = self.val_coeff * value_loss + policy_loss - self.entropy_coeff * entropy_reg

        return loss 


    def train(self, max_episodes, gamma, save_interval):

        total_rewards = np.zeros(max_episodes)
        progress = tqdm(range(self.start_episode, max_episodes))

        for episode in progress:

            reward, buffer = self.run_episode(episode)

            self.optim.zero_grad()
            loss = self.a2c_loss(buffer, gamma) 
            loss.backward()
            if self.max_grad_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optim.step()

            total_rewards[episode] = reward

            avg_reward_10 = total_rewards[max(0, episode-10):(episode+1)].mean()
            avg_reward_100 = total_rewards[max(0, episode-100):(episode+1)].mean()
            self.writer.add_scalar("perf/reward_t", reward, episode)
            self.writer.add_scalar("perf/avg_reward_10", avg_reward_10, episode)
            self.writer.add_scalar("perf/avg_reward_100", avg_reward_100, episode)
            self.writer.add_scalar("losses/total_loss", loss.item(), episode)
            if self.max_grad_norm > 0:
                self.writer.add_scalar("losses/grad_norm", grad_norm, episode)
            progress.set_description(f"Episode {episode}/{max_episodes} | Reward: {reward} | Last 10: {avg_reward_10:.4f} | Loss: {loss.item():.4f}")

            if (episode+1) % save_interval == 0:
                T.save({
                    "state_dict": self.agent.state_dict(),
                    "avg_reward_100": avg_reward_100,
                    'last_episode': episode,
                }, self.save_path.format(epi=episode+1) + ".pt")


    def test(self, num_episodes):
        progress = tqdm(range(num_episodes))
        self.env.reset_transition_count()
        self.agent.eval()
        total_rewards = np.zeros(num_episodes)
        for episode in progress:
            reward, _ = self.run_episode(episode)
            total_rewards[episode] = reward
            avg_reward = total_rewards[max(0, episode-10):(episode+1)].mean()            
            progress.set_description(f"Episode {episode}/{num_episodes} | Reward: {reward} | Last 10: {avg_reward:.4f}")

        self.env.plot(self.save_path.format(epi=self.seed))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="configs/ep_two_step.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    n_seeds = 1
    base_seed = 1111
    base_run_title = config["run-title"]
    for seed_idx in range(1, n_seeds + 1):
        config["run-title"] = base_run_title + f"_{seed_idx}"
        config["seed"] = base_seed * seed_idx
        
        exp_path = os.path.join(config["save-path"], config["run-title"])
        if not os.path.isdir(exp_path): 
            os.mkdir(exp_path)
        
        out_path = os.path.join(exp_path, os.path.basename(args.config))
        with open(out_path, 'w') as fout:
            yaml.dump(config, fout)

        print(f"> Running {config['run-title']}")
        trainer = Trainer(config)
        if config["train"]:
            trainer.train(config["task"]["train-episodes"], config["agent"]["gamma"], config["save-interval"])
        if config["test"]:
            trainer.test(config["task"]["test-episodes"])

    