"""
    A DND-based LSTM based on ...
    Ritter, et al. (2018).
    Been There, Done That: Meta-Learning with Episodic Recall.
    Proceedings of the International Conference on Machine Learning (ICML).
"""

import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from models.dnd import DND
from models.ep_lstm_cell import EpLSTMCell

class A2C_DND_LSTM(nn.Module):

    def __init__(self, 
            input_dim, 
            hidden_dim, 
            num_actions,
            dict_len,
            kernel='l2', 
            noise_idx=None,
    ):
        super(A2C_DND_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_idx = noise_idx

        # long-term memory 
        self.dnd = DND(dict_len, hidden_dim, kernel)

        # short-term memory
        self.ep_lstm = EpLSTMCell(
            input_size=input_dim,
            hidden_size=hidden_dim,
            noise_idx=noise_idx,
            noise_level=5
        )

        # intial states of LSTM
        self.h0 = nn.Parameter(T.randn(self.ep_lstm.Dh).float())
        self.c0 = nn.Parameter(T.randn(self.ep_lstm.Dh).float())

        # actor-critic networks
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # reset lstm parameters
        self.ep_lstm.reset_parameters_()
        # reset initial states
        T.nn.init.normal_(self.h0)
        T.nn.init.normal_(self.c0)
        # intialize actor and critic weights
        T.nn.init.orthogonal_(self.actor.weight, gain=0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight, gain=1.0)
        self.critic.bias.data.fill_(0)

    def forward(self, data, cue, mem_state):

        state, p_action, p_reward, timestep = data 
        x_t = T.cat((state, p_action, p_reward, timestep), dim=-1)

        if mem_state is None:
            mem_state = (self.h0, self.c0)

        m_t = self.dnd.get_memory(cue)
    
        _, (h_t, c_t) = self.ep_lstm(x_t, m_t, mem_state)

        # noisy_ht = h_t.clone() 
        # if self.noise_idx is not None:
        #     noise = T.randn(len(self.noise_idx))
        #     noisy_ht[:, self.noise_idx] = noisy_ht[:, self.noise_idx] + noise * 5

        action_dist = F.softmax(self.actor(h_t), dim=-1)
        value_estimate = self.critic(h_t)

        return action_dist, value_estimate, (h_t, c_t)

    def get_init_states(self):
        return (self.h0, self.c0)

    def turn_off_encoding(self):
        self.dnd.encoding_off = True

    def turn_on_encoding(self):
        self.dnd.encoding_off = False

    def turn_off_retrieval(self):
        self.dnd.retrieval_off = True

    def turn_on_retrieval(self):
        self.dnd.retrieval_off = False

    def reset_memory(self):
        self.dnd.reset_memory()

    def save_memory(self, mem_key, mem_val):
        self.dnd.save_memory(mem_key, mem_val)

    def retrieve_memory(self, query_key):
        return self.dnd.get_memory(query_key)

    def get_gates(self):
        return (
            np.array(self.ep_lstm.rt_gate).mean(axis=0),
            np.array(self.ep_lstm.it_gate).mean(axis=0),
            np.array(self.ep_lstm.ft_gate).mean(axis=0),
        )

    def get_all_mems(self):
        n_mems = len(self.dnd.keys)
        K = [self.dnd.keys[i] for i in range(n_mems)]
        V = [self.dnd.vals[i] for i in range(n_mems)]
        return K, V
