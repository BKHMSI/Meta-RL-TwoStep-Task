"""
    A DND-based LSTM based on ...
    Ritter, et al. (2018).
    Been There, Done That: Meta-Learning with Episodic Recall.
    Proceedings of the International Conference on Machine Learning (ICML).
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from models.DND import DND
from models.ep_lstm import EpLSTM

class A2C_DND_LSTM(nn.Module):

    def __init__(self, 
            input_dim, 
            hidden_dim, 
            num_actions,
            dict_len,
            kernel='l2', 
            bias=True
    ):
        super(A2C_DND_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias

        # long-term memory 
        self.dnd = DND(dict_len, hidden_dim, kernel)

        # short-term memory
        self.ep_lstm = EpLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # intial states of LSTM
        self.h0 = nn.Parameter(T.randn(1, 1, self.ep_lstm.hidden_size).float())
        self.c0 = nn.Parameter(T.randn(1, 1, self.ep_lstm.hidden_size).float())

        # actor-critic networks
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # reset lstm parameters
        self.ep_lstm.reset_parameters()
        # reset initial states
        T.nn.init.normal_(self.h0)
        T.nn.init.normal_(self.c0)
        # intialize actor and critic weights
        T.nn.init.orthogonal_(self.actor.weight, gain=0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight, gain=1.0)
        self.critic.bias.data.fill_(0)

    def forward(self, x_t, state):
        m_t = self.dnd.get_memory(x_t)
        h_t, (_, c_t) = self.ep_lstm((x_t, m_t), state)

        self.dnd.save_memory(x_t, c_t)

        action_dist = F.softmax(self.actor(h_t), dim=-1)
        value_estimate = self.critic(h_t)

        return action_dist, value_estimate, (h_t, c_t)
        
    def pick_action(self, action_distribution):
        """action selection by sampling from a multinomial.

        Parameters
        ----------
        action_distribution : 1d T.tensor
            action distribution, pi(a|s)

        Returns
        -------
        T.tensor(int), T.tensor(float)
            sampled action, log_prob(sampled action)

        """
        m = T.distributions.Categorical(action_distribution)
        a_t = m.sample()
        log_prob_a_t = m.log_prob(a_t)
        return a_t, log_prob_a_t

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

    def get_all_mems(self):
        n_mems = len(self.dnd.keys)
        K = [self.dnd.keys[i] for i in range(n_mems)]
        V = [self.dnd.vals[i] for i in range(n_mems)]
        return K, V
