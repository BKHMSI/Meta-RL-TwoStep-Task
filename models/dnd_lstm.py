"""
A DND-based LSTM based on ...
Ritter, et al. (2018).
Been There, Done That: Meta-Learning with Episodic Recall.
Proceedings of the International Conference on Machine Learning (ICML).
"""
import torch as T
import torch.nn as nn
from model.DND import DND
from model.A2C import A2C_linear


# constants
N_GATES = 4


class DNDLSTM(nn.Module):

    def __init__(
            self, input_dim, hidden_dim, output_dim,
            dict_len,
            kernel='l2', bias=True
    ):
        super(DNDLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        # input-hidden weights
        self.i2h = nn.Linear(input_dim, (N_GATES+1) * hidden_dim, bias=bias)
        # hidden-hidden weights
        self.h2h = nn.Linear(hidden_dim, (N_GATES+1) * hidden_dim, bias=bias)
        # dnd
        self.dnd = DND(dict_len, hidden_dim, kernel)
        #
        self.a2c = A2C_linear(hidden_dim, output_dim)
        # init
        self.reset_parameter()

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            if 'weight' in name:
                T.nn.init.orthogonal_(wts)
            elif 'bias' in name:
                T.nn.init.constant_(wts, 0)

    def forward(self, x_t, h, c):
        # unpack activity
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x_t = x_t.view(x_t.size(1), -1)
        # transform the input info
        Wx = self.i2h(x_t)
        Wh = self.h2h(h)
        preact = Wx + Wh
        # get all gate values
        gates = preact[:, : N_GATES * self.hidden_dim].sigmoid()
        # split input(write) gate, forget gate, output(read) gate
        f_t = gates[:, :self.hidden_dim]
        i_t = gates[:, self.hidden_dim:2 * self.hidden_dim]
        o_t = gates[:, 2*self.hidden_dim:3 * self.hidden_dim]
        r_t = gates[:, -self.hidden_dim:]
        # stuff to be written to cell state
        c_t_new = preact[:, N_GATES * self.hidden_dim:].tanh()
        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = T.mul(f_t, c) + T.mul(i_t, c_t_new)
        # retrieve memory
        m_t = self.dnd.get_memory(x_t).tanh()
        # gate the memory; in general, can be any transformation of it
        c_t = c_t + T.mul(r_t, m_t)
        # get gated hidden state from the cell state
        h_t = T.mul(o_t, c_t.tanh())
        # take a episodic snapshot
        self.dnd.save_memory(x_t, c_t)
        # policy
        pi_a_t, v_t = self.a2c.forward(h_t)
        # pick an action
        a_t, prob_a_t = self.pick_action(pi_a_t)
        # reshape data
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        # fetch activity
        output = [a_t, prob_a_t, v_t, h_t, c_t]
        cache = [f_t, i_t, o_t, r_t, m_t]
        return output, cache

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

    def get_init_states(self, scale=.1):
        h_0 = T.randn(1, 1, self.hidden_dim) * scale
        c_0 = T.randn(1, 1, self.hidden_dim) * scale
        return h_0, c_0

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
