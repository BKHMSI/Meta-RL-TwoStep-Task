
import numpy as np

# constants
S_1 = 0
S_2 = 1
S_3 = 2
N_STATES = 3

class TwoStepTask:
    def __init__(self, config):
        # start state
        self.state = S_1

        # feature dimension: (onehot state, action, timestep, reward)
        self.feat_size = 6
        
        # defines the stage with the highest expected reward, initially random
        self.highest_reward_state = np.random.choice([S_2, S_3])
        
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
        if np.random.uniform() < switch_p and self.state == S_1:
            self.highest_reward_state = S_2 if (self.highest_reward_state == S_3) else S_3
            
    def reward_probs(self):
        """probability of reward of states S_2 and S_3, in the form [[p, 1-p], [1-p, p]]
        """
        if (self.highest_reward_state == S_2):
            r_prob = 0.9
        else:
            r_prob = 0.1
        
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
        self.last_state = None
        
        self.state = S_1
        
        return self.encode_state()
        
    def _step(self, action):
        self.timestep += 1
        self.last_state = self.state
        
        if self.state == S_1:
            # get reward
            reward = 0
            # update stage
            self.state = S_2 if (np.random.uniform() < self.transitions[action][0]) else S_3

            # keep track of stay probability after first action
            if self.last_action is not None:    
                self.update_stay_prob(action)

            self.last_action = action
            # book-keeping for plotting
            self.last_is_common = self.is_common(action, self.state-1)
            
        else:# case S_2 or S_3
            # get probability of reward in stage
            r_prob = 0.9 if (self.highest_reward_state == self.state) else 0.1
            # get reward
            reward = 1 if np.random.uniform() < r_prob else 0
            # update stage
            self.state = S_1
            # book-keeping for plotting
            self.last_is_rewarded = reward

        new_state = self.encode_state()

        done = self.timestep >= self.num_trials
    
        return new_state, reward, done, self.timestep
    
    def step(self, action):
        # stage 1
        obs, _, _, _ = self._step(action)
        # stage 2
        _, reward, done,_ = self._step(action)
        return obs, reward, done, self.timestep