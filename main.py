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
from utils import save_data, load_data

Rollout = namedtuple('Rollout',
                        ('state', 'action', 'reward', 'timestep', 'done', 'value'))

class Trainer: 
    def __init__(self, config):
        self.device = 'cpu'
        self.seed = config["seed"]

        T.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        T.random.manual_seed(config["seed"])

        self.env = TwoStepTask(config["task"])  
        self.model = A2C_LSTM(config["a2c"], self.env.feat_size, self.env.num_actions).to(self.device)

        self.optim = T.optim.RMSprop(self.model.parameters(), lr=config["a2c"]["lr"])

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
            self.model.load_state_dict(T.load(self.save_path.format(epi=self.start_episode))["state_dict"])

    def a2c_loss(self, buffer, gamma, bootstrap_value=0):


        def discount(data):
            data = data.numpy()
            return T.tensor(scipy.signal.lfilter([1], [1, -gamma], data[::-1], axis=0)[::-1].copy(), device=self.device)

        batch = Rollout(*zip(*buffer))

        dones = T.tensor(batch.done, device=self.device)
        states = T.tensor(batch.state, device=self.device).float()
        values = T.tensor(batch.value, device=self.device)
        rewards = T.tensor(batch.reward, device=self.device).float()

        timesteps = T.tensor(batch.timestep, device=self.device).unsqueeze(-1).float()
        bootstrap = T.tensor([bootstrap_value], device=self.device).float()

        prev_rewards = T.tensor([0] + list(batch.reward[:-1]), device=self.device).unsqueeze(-1).float()
        prev_actions = T.tensor([[0,0]] + list(batch.action[:-1]), device=self.device).float()
        action_onehot = T.tensor(batch.action, device=self.device)

        # The advantage function uses "Generalized Advantage Estimation"
        dones_plus = T.cat((dones, T.tensor([True], device=self.device)), dim=-1)
        rewards_plus = T.cat((rewards, bootstrap), dim=-1) * ~dones_plus
        discounted_rewards = discount(rewards_plus)[:-1]
        value_plus = T.cat((values, bootstrap), dim=-1)
        advantages = rewards + gamma * value_plus[1:] * (~dones) - value_plus[:-1]
        advantages = discount(advantages)
                
        policy, values_out, _ = self.model((
            states,
            prev_actions,
            prev_rewards,
            timesteps,
            None
        ))

        logits = (policy * action_onehot).sum(2)
        value_loss = 0.5 * (discounted_rewards - values_out).pow(2).mean()
        policy_loss = - (T.log(logits) * advantages).mean()
        entropy_reg = - (policy * T.log(policy)).mean()

        loss = self.val_coeff * value_loss + policy_loss - self.entropy_coeff * entropy_reg

        return loss 

    def run_episode(self, episode):
        done = False 
        total_reward, total_entropy = 0, 0
        p_action, p_reward, timestep = [0,0], 0, 0

        state = self.env.reset()
        mem_state = self.model.init_state(device=self.device)

        buffer = []
        while not done:

            # switch reward contingencies at the beginning of each episode with probability p
            self.env.possible_switch(switch_p=self.switch_p)

            # sample action using model
            action_dist, val_estimate, mem_state = self.model((
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

            # ('state', 'action', 'reward', 'timestep', 'done', 'value'))
            buffer += [Rollout(
                state, 
                action_onehot,
                reward,
                timestep,
                done,
                val_estimate
            )]

            state = new_state
            p_reward = reward
            p_action = action_onehot

            total_reward += reward

        return total_reward, buffer


    def train(self, max_episodes, gamma, save_interval):

        total_rewards = np.zeros(max_episodes)
        progress = tqdm(range(self.start_episode, max_episodes))
        for episode in progress:

            self.model.eval()
            reward, buffer = self.run_episode(episode)

            self.model.train()
            self.optim.zero_grad()
            loss = self.a2c_loss(buffer, gamma) 
            loss.backward()
            if self.max_grad_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optim.step()

            total_rewards[episode] = reward

            avg_reward = total_rewards[max(0, episode-10):(episode+1)].mean()
            self.writer.add_scalar("perf/reward_t", reward, episode)
            self.writer.add_scalar("perf/avg_reward_10", avg_reward, episode)
            self.writer.add_scalar("losses/total_loss", loss.item(), episode)
            if self.max_grad_norm > 0:
                self.writer.add_scalar("losses/grad_norm", grad_norm, episode)
            progress.set_description(f"Episode {episode}/{max_episodes} | Reward: {reward} | Last 10: {avg_reward:.4f} | Loss: {loss.item():.4f}")

            if (episode+1) % save_interval == 0:
                self.env.plot(self.save_path.format(epi=episode+1))
                T.save({
                    "state_dict": self.model.state_dict(),
                    "avg_reward": avg_reward,
                    'last_episode': episode,
                }, self.save_path.format(epi=episode+1) + ".pt")


    def test(self, num_episodes):
        progress = tqdm(range(num_episodes))
        self.env.reset_transition_count()
        self.model.eval()
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

    exp_path = os.path.join(config["save-path"], config["run-title"])
    if not os.path.isdir(exp_path): 
        os.mkdir(exp_path)
    
    out_path = os.path.join(exp_path, os.path.basename(args.config))
    with open(out_path, 'w') as fout:
         yaml.dump(config, fout)

    print(f"> Running {config['run-title']}")
    trainer = Trainer(config)
    trainer.train(config["task"]["train-episodes"], config["a2c"]["gamma"], config["save-interval"])
    trainer.test(config["test"]["test-episodes"])

    