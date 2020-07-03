
import numpy as np

# constants
S_0 = 0
S_1 = 1
S_2 = 2
N_STATES = 3

class TwoStepTask:
    def __init__(self, config):
        # start state
        self.state = S_0

        # feature dimension: (onehot state, onehot action, timestep, reward)
        self.feat_size = 7
        
        # defines the stage with the highest expected reward, initially random
        self.highest_reward_state = np.random.choice([S_1, S_2])
        
        self.num_actions = 2
        self.r_prob = config["reward-prob"]
        self.num_trials = config["trials-per-epi"]
        
        # initialization of plotting variables
        common_prob = config["common-prob"]
        self.transitions = np.array([
            [common_prob, 1-common_prob],
            [1-common_prob, common_prob]
        ])

        # keep track of stay probability 
        self.transition_count = np.zeros((2,2,2))
        self.reset()
        
   
    def encode_state(self):
        return np.eye(N_STATES)[self.state]

    def possible_switch(self, switch_p=0.025):
        """switch reward contingencies at the beginning of each trial with some probability 
        
        Parameters
        ----------
            switch_p : int
                probability of switching 
        """
        if np.random.uniform() < switch_p and self.state == S_0:
            self.highest_reward_state = S_1 if (self.highest_reward_state == S_2) else S_2
            
    def reward_probs(self):
        """probability of reward of states S_1 and S_2, in the form [[p, 1-p], [1-p, p]]
        """
        r_prob = self.r_prob if self.highest_reward_state == S_1 else 1-self.r_prob
        rewards = np.array([
            [r_prob, 1-r_prob],
            [1-r_prob, r_prob]
        ])
        return rewards
            
    def is_common(self, action, state):
        return self.transitions[action][state] >= 0.5
        
    def update_stay_prob(self, action):
        self.transition_count[
            self.last_is_rewarded,
            self.last_is_common,
            self.last_action
        ] += 1
 
    def compute_stay_prob(self):
        action_count = self.transition_count.sum(axis=2)
        return self.transition_count / action_count[:, :, np.newaxis]

    def reset(self):
        self.timestep = 0
        
        # for the two-step task plots
        self.last_is_common = None
        self.last_is_rewarded = None
        self.last_action = None
        
        self.state = S_0
        
        return self.encode_state()

    def _stage_1(self, action):
        # act and update stage
        self.state = S_1 if (np.random.uniform() < self.transitions[action][0]) else S_2

        # keep track of stay probability after first action
        if self.last_action is not None:    
            self.update_stay_prob(action)

        self.last_action = action
        self.last_is_common = self.is_common(action, self.state-1) 

        new_state = self.encode_state()
        return new_state

    def _stage_2(self):
        # get probability of reward in stage
        r_prob = self.r_prob if (self.highest_reward_state == self.state) else 1-self.r_prob
        # get reward
        reward = 1 if np.random.uniform() < r_prob else 0
        # update stage
        self.state = S_0
        # book-keeping for plotting
        self.last_is_rewarded = reward 
        return reward

    def step(self, action):
        """two-step task trial 
        
        Parameters
        ----------
            action : int
                action to perform in stage 1 
        """
        self.timestep += 1

        state  = self._stage_1(action)
        reward = self._stage_2()

        done = self.timestep >= self.num_trials

        return state, reward, done, self.timestep