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

from models.a2c_lstm import A2C_LSTM
from tasks.two_step import TwoStepTask

Rollout = namedtuple('Rollout',
                        ('state', 'action', 'reward', 'timestep', 'done', 'policy', 'value'))

class Trainer: 
    def __init__(self, config):
        self.device = 'cpu'
        self.seed = config["seed"]

        T.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        T.random.manual_seed(config["seed"])

        self.env = TwoStepTask(config["task"])  
        self.agent = A2C_LSTM(config["a2c"], self.env.feat_size, self.env.num_actions).to(self.device)

        self.optim = T.optim.RMSprop(self.agent.parameters(), lr=config["a2c"]["lr"])

        self.val_coeff = config["a2c"]["value-loss-weight"]
        self.entropy_coeff = config["a2c"]["entropy-weight"]
        self.max_grad_norm = config["a2c"]["max-grad-norm"]
        self.switch_p = config["task"]["swtich-prob"]
        self.start_episode = 0

        self.writer = SummaryWriter(log_dir=os.path.join("logs", config["run-title"]))
        self.save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}")

        if config["resume"]:
            print("> Loading Checkpoint")
            self.start_episode = config["start-episode"]
            self.agent.load_state_dict(T.load(self.save_path.format(epi=self.start_episode) + ".pt")["state_dict"])

    def run_episode(self, episode):
        done = False
        total_reward = 0
        p_action, p_reward, timestep = [0,0], 0, 0

        state = self.env.reset()
        mem_state = self.agent.get_init_states()

        buffer = []
        while not done:

            # switch reward contingencies at the beginning of each episode with probability p
            self.env.possible_switch(switch_p=self.switch_p)

            # sample action using model
            action_dist, val_estimate, mem_state = self.agent((
                T.tensor([state], device=self.device).float(), 
                T.tensor([p_action], device=self.device).float(),  
                T.tensor([[p_reward]], device=self.device).float(),  
                T.tensor([[timestep]], device=self.device).float(), 
                mem_state
            ))

            action_cat = T.distributions.Categorical(action_dist.squeeze())
            action = action_cat.sample()
            action_onehot = np.eye(2)[action]

            # take action and observe result
            new_state, reward, done, timestep = self.env.step(int(action))

            # ('state', 'action', 'reward', 'timestep', 'done', 'policy', 'value')
            buffer += [Rollout(
                state, 
                action_onehot,
                reward,
                timestep,
                done,
                action_dist,
                val_estimate
            )]

            state = new_state
            p_reward = reward
            p_action = action_onehot

            total_reward += reward

        # boostrap final observation 
        _, val_estimate, _ = self.agent((
            T.tensor([state], device=self.device).float(), 
            T.tensor([p_action], device=self.device).float(),  
            T.tensor([[p_reward]], device=self.device).float(),  
            T.tensor([[timestep]], device=self.device).float(), 
            mem_state
        ))

        buffer += [Rollout(None, None, None, None, None, None, val_estimate)]

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

        policy = T.cat(batch.policy[:-1], dim=1).squeeze().to(self.device)
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
    parser.add_argument('-c', '--config',  type=str, default="configs/two_step.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    n_seeds = 8
    base_run_title = config["run-title"]
    for seed_idx in range(1, n_seeds + 1):
        config["run-title"] = base_run_title + f"_{seed_idx}"
        config["seed"] = 1111 * seed_idx
        
        exp_path = os.path.join(config["save-path"], config["run-title"])
        if not os.path.isdir(exp_path): 
            os.mkdir(exp_path)
        
        out_path = os.path.join(exp_path, os.path.basename(args.config))
        with open(out_path, 'w') as fout:
            yaml.dump(config, fout)

        print(f"> Running {config['run-title']}")
        trainer = Trainer(config)
        if config["train"]:
            trainer.train(config["task"]["train-episodes"], config["a2c"]["gamma"], config["save-interval"])
        if config["test"]:
            trainer.test(config["task"]["test-episodes"])

    