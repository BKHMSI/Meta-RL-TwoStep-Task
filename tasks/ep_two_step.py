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

    def _binary2int(self, binary):
        return binary.dot(1 << np.arange(binary.shape[-1] - 1, -1, -1))

    def _generate_context(self):
        return np.random.randint(2, size=self.ctx_len)

    def _generate_uncue(self):
        return np.ones(self.ctx_len) * -1

    def _stage_2(self, context):
        # reward based on which part in the trial
        if self.timestep < 50:
            r_prob = self.r_prob if (self.highest_reward_state == self.state) else 1-self.r_prob
            reward = 1 if np.random.uniform() < r_prob else 0
        elif self.timestep < 75:
            reward = self.memory_1[context]
        else:
            reward = self.memory_2[context]

        # update stage
        self.state = S_0
        # book-keeping for plotting
        self.last_is_rewarded = reward 
        return reward

    def step(self, action, cue):

        self.timestep += 1

        # take action and go to next stage
        state = self._stage_1(action)

        # observe context
        if self.timestep < 50:
            context = self._generate_context()
        elif self.timestep < 75:
            context = random.sample(self.memory_1.keys(), k=1)
        else:
            context = random.sample(self.memory_2.keys(), k=1)

        ctx_int = self._binary2int(context)

        reward = self._stage_2(ctx_int)

        if self.timestep < 25:
            self.memory_1[ctx_int] = reward
        elif self.timestep < 50:
            self.memory_2[ctx_int] = reward
        elif self.timestep < 75:
            pass 
        else:
            pass 
            
        done = self.timestep >= self.num_trials



        return state, reward, done, context, self.timestep
