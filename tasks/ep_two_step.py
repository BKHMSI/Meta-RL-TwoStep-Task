# from tasks.two_step import TwoStepTask

import random 
import numpy as np
from tasks.two_step import TwoStepTask

# constants
S_0 = 0
S_1 = 1
S_2 = 2
N_STATES = 3


class EpTwoStepTask(TwoStepTask):
    def __init__(self, config):
        self.ctx_len = config["context-len"]
        self.memory_1 = {} # maps context to reward for trials 1-25
        self.memory_2 = {} # maps context to reward for trials 26-50
        super().__init__(config)

    def reset(self):
        self.timestep = 0
        
        # for the two-step task plots
        self.last_is_common = None
        self.last_is_rewarded = None
        self.last_action = None

        # clear memories
        self.memory_1 = {}
        self.memory_2 = {}
        
        self.state = S_0
        
        state = self.encode_state()
        return state 

    def _generate_context(self):
        return np.random.randint(2, size=self.ctx_len)

    def _generate_uncue(self):
        return np.ones(self.ctx_len) * -1

    def _stage_2(self):
        # reward based on which part in the trial
        r_prob = self.r_prob if (self.highest_reward_state == self.state) else 1-self.r_prob
        reward = 1 if np.random.uniform() < r_prob else 0
        return reward

    def get_trial(self):
        return "uncued" if self.timestep < 50 else "cued"

    def get_cue(self):
        if self.timestep < 50:
            return self._generate_uncue()
        elif self.timestep < 75:
            rand_cue = np.random.choice(list(self.memory_1.keys()))
        else:
            rand_cue = np.random.choice(list(self.memory_2.keys()))
        return _int2binary(rand_cue, self.ctx_len)

    def step(self, action, cue):
        # take action and go to next stage
        state = self._stage_1(action)

        if self.timestep < 50:
            reward = self._stage_2()
            context = self._generate_context()
            ctx_int = _binary2int(context)
            if self.timestep < 25:
                self.memory_1[ctx_int] = reward
            else:
                self.memory_2[ctx_int] = reward
        elif self.timestep < 75:
            reward = self.memory_1[_binary2int(cue)]
            context = cue 
        else:
            reward = self.memory_2[_binary2int(cue)]
            context = cue 

        # update stage
        self.state = S_0
        # book-keeping for plotting
        self.last_is_rewarded = reward 
            
        self.timestep += 1
        done = self.timestep >= self.num_trials
        

        return state, reward, done, self.timestep, context


"""helpers"""

def _binary2int(binary):
    return (binary * 2**np.arange(binary.shape[0]-1, -1, -1)).sum()

def _int2binary(decimal, length=10):
    return np.array([int(x) for x in format(decimal, f'#0{length+2}b')[2:]])