import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def ortho_init(weights, scale=1.):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(weights.size())
    flat_shape = shape[1], shape[0]

    a = T.tensor(np.random.normal(0., 1., flat_shape))

    u, _, v = T.svd(a)
    t = u if u.shape == flat_shape else v
    t = t.transpose(1,0).reshape(shape).float()

    return scale * t

class A2C_LSTM(nn.Module):

    def __init__(self, config, input_dim, num_actions):
        super(A2C_LSTM, self).__init__()
        
        self.actor = nn.Linear(config["mem-units"], num_actions)
        self.critic = nn.Linear(config["mem-units"], 1)
        self.working_memory = nn.LSTM(input_dim, config["mem-units"])

        self.h0 = nn.Parameter(T.randn(1, 1, self.working_memory.hidden_size).float())
        self.c0 = nn.Parameter(T.randn(1, 1, self.working_memory.hidden_size).float())

        # intialize actor and critic weights
        self.actor.weight.data = ortho_init(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        self.critic.weight.data = ortho_init(self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)
        
    def forward(self, data):
        state, p_action, p_reward, timestep, mem_state = data 
        p_input = T.cat((state, p_action, p_reward, timestep), dim=-1)

        if mem_state is None:
            mem_state = (self.h0, self.c0)
    
        h_t, mem_state = self.working_memory(p_input.unsqueeze(1), mem_state)

        action_dist = F.softmax(self.actor(h_t), dim=-1)
        value_estimate = self.critic(h_t)

        return action_dist, value_estimate, mem_state

    def get_init_states(self):
        return (self.h0, self.c0)