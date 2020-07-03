import os
import yaml
import random 
import argparse
import numpy as np

import torch as T
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.a2c_lstm import A2C_LSTM
from tasks.two_step import TwoStepTask

from tqdm import tqdm 
from collections import namedtuple

Rollout = namedtuple('Rollout',
                        ('state', 'action', 'reward', 'done', 'log_prob_a', 'value'))

class Trainer: 
    def __init__(self, config):
        self.device = 'cpu'

        self.env = TwoStepTask(config["Task"])  
        self.model = A2C_LSTM(config["A2C"], self.env.feat_size, self.env.num_actions).to(self.device)

        self.optim = T.optim.AdamW(self.model.parameters(), lr=config["A2C"]["lr"], weight_decay=config["A2C"]["weight-decay"])

        self.val_coeff = config["A2C"]["val-loss-weight"]
        self.entropy_coeff = config["A2C"]["entropy-weight"]
        self.max_grad_norm = config["A2C"]["max-grad-norm"]
        self.switch_p = config["Task"]["swtich-prob"]

        self.writer = SummaryWriter(log_dir=os.path.join("logs", config["run-title"]))
        self.save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}.pt")

    def a2c_loss(self, buffer, gamma, entropy, bootstrap_value=0):


        def discount(rewards, masks):
            d_reward = T.tensor([gamma**i * rewards[i] * masks[i] for i in range(len(rewards))], device=self.device)
            return T.flip(T.cumsum(T.flip(d_reward, [0]), 0), [0])

        batch = Rollout(*zip(*buffer))

        dones = T.tensor(batch.done, device=self.device)
        values = T.tensor(batch.value, device=self.device)
        rewards = T.tensor(batch.reward, device=self.device).float()
        log_probs = T.tensor(batch.log_prob_a, device=self.device)
        bootstrap = T.tensor([bootstrap_value], device=self.device).float()

        # here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # the advantage function uses "Generalized Advantage Estimation"
        dones_plus = T.cat((dones, T.tensor([True], device=self.device)), dim=-1)
        rewards_plus = T.cat((rewards, bootstrap), dim=-1)
        discounted_rewards = discount(rewards_plus, ~dones_plus)[:-1]
        value_plus = T.cat((values, bootstrap), dim=-1)
        advantages = rewards + gamma * value_plus[1:] * (~dones) - value_plus[:-1]
        advantages = discount(advantages, ~dones)

        # calculate losses
        value_loss = 0.5 * (discounted_rewards - values).sum()
        policy_loss = -(log_probs * advantages).sum()

        loss = self.val_coeff * value_loss + policy_loss - self.entropy_coeff * entropy

        return loss 


    def run_episode(self, episode):
        done = False 
        total_reward, total_entropy = 0, 0
        p_action, p_reward, timestep = 0, 0, 0

        state = self.env.reset()
        mem_state = self.model.init_state(device=self.device)

        buffer = []
        while not done:

            # switch reward contingencies at the beginning of each episode with probability p
            self.env.possible_switch(switch_p=self.switch_p)

            # sample action using model
            action_dist, val_estimate, mem_state = self.model((
                T.tensor(state,    device=self.device).float(), 
                T.tensor([p_reward], device=self.device).float(),  
                T.tensor([p_action], device=self.device).float(),  
                T.tensor([timestep], device=self.device).float(), 
                mem_state
            ))

            action_cat = T.distributions.Categorical(action_dist)
            action = action_cat.sample()
            entropy = action_cat.entropy().mean()
            log_prob_a = action_cat.log_prob(action)

            # take action and observe result
            new_state, reward, done, timestep = self.env.step(int(action))

            buffer += [Rollout(
                state, 
                reward,
                action,
                done,
                log_prob_a,
                val_estimate.item()
            )]

            state = new_state
            p_reward = reward
            p_action = action 

            total_reward += reward
            total_entropy += entropy

        return total_reward, total_entropy, buffer


    def train(self, max_episodes, gamma):

        total_rewards = np.zeros(max_episodes)

        progress = tqdm(range(max_episodes))
        for episode in progress:

            reward, entropy, buffer = self.run_episode(episode)
            
            self.optim.zero_grad()
            loss = self.a2c_loss(buffer, gamma, entropy) 
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optim.step()

            total_rewards[episode] = reward

            avg_reward = total_rewards[max(0, episode-100):(episode+1)].mean()
            self.writer.add_scalar("rewards/reward_t", reward, episode)
            self.writer.add_scalar("rewards/avg_reward_100", avg_reward, episode)
            # print(f"Episode {episode}/{max_episodes} | Reward: {reward} | Last 100: {avg_reward}", end="\r")

            progress.set_description(f"Episode {episode}/{max_episodes} | Reward: {reward} | Last 100: {avg_reward}")

            if (episode+1) % 2500 == 0:
                T.save({
                    "state_dict": self.model.state_dict(),
                    "avg_reward": avg_reward,
                }, self.save_path.format(epi=episode+1))



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

    trainer = Trainer(config)
    trainer.train(config["Task"]["train-episodes"], config["A2C"]["gamma"])

    