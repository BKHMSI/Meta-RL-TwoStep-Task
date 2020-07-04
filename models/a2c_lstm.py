import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F


def ortho_weights(shape, scale=1.):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return T.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return T.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))

def normalized_columns_initializer(weights, std=1.0):
    out = T.randn(weights.size())
    out *= std / T.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

class A2C_LSTM(nn.Module):

    def __init__(self, config, input_dim, num_actions):
        super(A2C_LSTM, self).__init__()
        
        self.actor = nn.Linear(config["mem-units"], num_actions)
        self.critic = nn.Linear(config["mem-units"], 1)
        self.working_memory = nn.LSTMCell(input_dim, config["mem-units"])

        # intialize actor and critic weights
        self.actor.weight.data = normalized_columns_initializer(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        self.critic.weight.data = normalized_columns_initializer(self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)
        
    def forward(self, data):
        state, p_action, p_reward, timestep, mem_state = data 
        p_input = T.cat((state, p_action, p_reward, timestep), dim=-1)
    
        h_t, c_t = self.working_memory(p_input.unsqueeze(0), mem_state)
        action_dist = F.softmax(self.actor(h_t), dim=-1)
        value_estimate = self.critic(h_t)

        return action_dist, value_estimate, (h_t, c_t)

    def init_state(self, device):
        h0 = T.zeros(1, self.working_memory.hidden_size).float().to(device)
        c0 = T.zeros(1, self.working_memory.hidden_size).float().to(device)
        return (h0, c0)