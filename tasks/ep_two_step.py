# from tasks.two_step import TwoStepTask

import random 
import numpy as np
from two_step import TwoStepTask

# constants
S_0 = 0
S_1 = 1
S_2 = 2
N_STATES = 3


class EpTwoStepTask(TwoStepTask):
    def __init__(self, config):
        super().__init__(config)

        self.ctx_len = config["context-length"]
        self.memory_1 = {} # maps context to reward for trials 1-25
        self.memory_2 = {} # maps context to reward for trials 26-50

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
        
        uncue = self._generate_uncue()
        state = self.encode_state()
        return state, uncue 

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
            return _int2binary(np.random.choice(list(self.memory_1.keys())))
        else:
            return _int2binary(np.random.choice(list(self.memory_2.keys())))

    def step(self, action, cue):

        self.timestep += 1

        # take action and go to next stage
        state = self._stage_1(action)

        if self.timestep < 50:
            reward = self._stage_2()
            context = self._generate_context()
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
            
        done = self.timestep >= self.num_trials

        return state, reward, done, context, self.timestep


"""helpers"""

def _binary2int(binary):
    return binary.dot(1 << np.arange(binary.shape[-1] - 1, -1, -1))

def _int2binary(decimal):
    return np.array([int(x) for x in format(decimal, '#012b')[2:]])