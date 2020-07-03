import torch as T
import torch.nn as nn
import torch.nn.functional as F


class A2C_LSTM(nn.Module):

    def __init__(self, config, input_dim, num_actions):
        super(A2C_LSTM, self).__init__()
        
        self.actor = nn.Linear(config["mem-units"], num_actions)
        self.critic = nn.Linear(config["mem-units"], 1)

        self.working_memory = nn.LSTMCell(input_dim, config["mem-units"])

    def forward(self, data):
        state, p_reward, p_action, timestep, mem_state = data 
        p_input = T.cat((state, p_reward, p_action, timestep), dim=-1)
        
        h_t, c_t = self.working_memory(p_input.unsqueeze(0), mem_state)
        action_dist = F.softmax(self.actor(h_t), dim=-1)
        value_estimate = self.critic(h_t)

        return action_dist, value_estimate, (h_t, c_t)

    def init_state(self, device):
        h0 = T.zeros(1, self.working_memory.hidden_size).float().to(device)
        c0 = T.zeros(1, self.working_memory.hidden_size).float().to(device)
        return (h0, c0)

        

