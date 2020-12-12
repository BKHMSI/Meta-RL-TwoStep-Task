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
    def __init__(self, config, noise_idx = None):
        self.device = 'cpu'
        self.seed = config["seed"]
        self.mode = config["mode"]

        T.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        T.random.manual_seed(config["seed"])

        self.env = EpTwoStepTask(config["task"])  
        self.agent = A2C_DND_LSTM(
            self.env.feat_size,
            config["agent"]["mem-units"], 
            self.env.num_actions,
            config["agent"]["dict-len"],
            config["agent"]["dict-kernel"],
            noise_idx
        ).to(self.device)

        self.optim = T.optim.RMSprop(self.agent.parameters(), lr=config["agent"]["lr"])

        self.val_coeff = config["agent"]["value-loss-weight"]
        self.entropy_coeff = config["agent"]["entropy-weight"]
        self.max_grad_norm = config["agent"]["max-grad-norm"]
        self.switch_p = config["task"]["swtich-prob"]
        self.start_episode = 0

        self.writer = SummaryWriter(log_dir=os.path.join("logs_ep", config["run-title"]))
        self.save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}")

        print("> Loading Checkpoint")
        self.start_episode = config["start-episode"]
        self.agent.load_state_dict(T.load(self.save_path.format(epi=self.start_episode) + ".pt")["state_dict"])

    def run_episode(self, episode):
        done = False
        total_reward = 0
        p_action, p_reward, timestep = [0,0], 0, 0

        self.agent.reset_memory()
        self.agent.turn_on_encoding()

        state = self.env.reset()
        (h_tm1, c_tm1) = self.agent.get_init_states()

        buffer = []

        while not done:

            # switch reward contingencies at the beginning of each episode with probability p
            self.env.possible_switch(switch_p=self.switch_p)

            if self.env.trial == "cued" and self.mode == "episodic":
                self.agent.turn_on_retrieval()
            else:
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

            action_dist, values, (h_t, c_t) = self.agent(x_t, cue, (h_tm1, c_tm1))
            
            action_cat = T.distributions.Categorical(action_dist.squeeze())
            action = action_cat.sample()
            action_onehot = np.eye(2)[action]

            # take action and observe result
            new_state, reward, done, timestep, context = self.env.step(int(action), cue.cpu().numpy())

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
            c_tm1 = c_t
            h_tm1 = h_t 

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

    def test(self, num_episodes):
        progress = tqdm(range(num_episodes))
        self.env.reset_transition_count()
        self.agent.eval()
        total_rewards = np.zeros(num_episodes)
        rt_list = []
        for episode in progress:
            reward, _ = self.run_episode(episode)
            rt, _, _ = self.agent.get_gates()
            self.agent.ep_lstm.reset_gate_monitor()
            rt_list += [rt]
            total_rewards[episode] = reward
            avg_reward = total_rewards[max(0, episode-10):(episode+1)].mean()            
            progress.set_description(f"Episode {episode}/{num_episodes} | Reward: {reward} | Last 10: {avg_reward:.4f}")

        if self.mode == "incremental":
            self.env.plot(self.save_path.format(epi=self.seed) + "_uncued", self.env.transition_count_uncued, "Incremental Uncued", y_lim=0)
            self.env.plot(self.save_path.format(epi=self.seed) + "_cued", self.env.transition_count_cued, "Incremental Cued", y_lim=0)
        elif self.mode == "episodic":
            self.env.plot(self.save_path.format(epi=self.seed) + "_episodic", self.env.transition_count_episodic, "Episodic", y_lim=0)

        return rt_list, self.env.total_reward_cued / (num_episodes*50), self.env.total_reward_uncued / (num_episodes*50)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="configs/ep_two_step.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    n_seeds = config["n-seeds"]
    base_seed = config["seed"]
    base_run_title = config["run-title"]
    threshold = 0.2

    reward_cued = np.zeros(n_seeds)
    reward_uncued = np.zeros(n_seeds)

    for seed_idx in range(1, n_seeds + 1):
        config["run-title"] = base_run_title + f"_{seed_idx}"
        config["seed"] = base_seed * seed_idx
        
        exp_path = os.path.join(config["save-path"], config["run-title"])
        if not os.path.isdir(exp_path): 
            os.makedirs(exp_path)
        
        out_path = os.path.join(exp_path, os.path.basename(args.config))
        with open(out_path, 'w') as fout:
            yaml.dump(config, fout)

        print(f"> Running {config['run-title']}")
        trainer = Trainer(config)
        rt, _, _ = trainer.test(config["task"]["test-episodes"])
        rt = np.array(rt).mean(axis=0)
        rt_select = np.arange(config["agent"]["mem-units"])[rt >= 1-threshold]

        print(f"{len(rt_select)}/{len(rt)}")


        # trainer = Trainer(config, noise_idx=rt_select)
        # _, reward_cued[seed_idx-1], reward_uncued[seed_idx-1] = trainer.test(config["task"]["test-episodes"])


    save_path = os.path.join(config["save-path"], "reward_cued.npy")
    np.save(save_path, reward_cued)

    save_path = os.path.join(config["save-path"], "reward_uncued.npy")
    np.save(save_path, reward_uncued)

    