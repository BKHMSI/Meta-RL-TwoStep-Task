
import numpy as np
import matplotlib.pyplot as plt 

# constants
S_0 = 0
S_1 = 1
S_2 = 2
N_STATES = 3

class TwoStepTask:
    def __init__(self, config):
        # start state
        self.state = S_0

        # feature dimension: (onehot state [3], onehot action [2], timestep [1], reward [1])
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
            
    def is_common(self, action, state):
        return self.transitions[action][state] >= 0.5
        
    def update_stay_prob(self, action):
        self.transition_count[
            int(not self.last_is_rewarded),
            int(not self.last_is_common),
            int(not (self.last_action == action))
        ] += 1
 
    def compute_stay_prob(self, transition_count):
        # stay_prob[r,c,a] = P[r,c,a] / (P[r,c,a] + P[r,c,~a]) 
        action_count = transition_count.sum(axis=-1)
        return transition_count / action_count[:, :, np.newaxis]

    def plot(self, save_path, transition_count=None, title="Two-Step Task", y_lim=0.5):
        _, ax = plt.subplots()

        ax.set_ylim([y_lim, 1.0])
        ax.set_ylabel('Stay Probability')
        ax.set_title(title)

        if transition_count is None:
            transition_count = self.transition_count
        
        stay_probs = self.compute_stay_prob(transition_count)
        
        common = [stay_probs[0,0,0], stay_probs[1,0,0]]
        uncommon = [stay_probs[0,1,0], stay_probs[1,1,0]]
        
        ax.set_xticks([1.5,3.5])
        ax.set_xticklabels(['Rewarded', 'Unrewarded'])
        
        c = plt.bar([1,3], common, color='b', width=0.5)
        uc = plt.bar([2,4], uncommon, color='r', width=0.5)
        ax.legend( (c[0], uc[0]), ('Common', 'Uncommon') )
        plt.savefig(save_path + ".png")
        np.save(save_path + ".npy", stay_probs)
      

    def reset(self):
        self.timestep = 0
        
        # for the two-step task plots
        self.last_is_common = None
        self.last_is_rewarded = None
        self.last_action = None
        
        self.state = S_0
        
        return self.encode_state()

    def reset_transition_count(self):
        self.transition_count = np.zeros((2,2,2))

    def _stage_1(self, action):
        # act and update stage
        self.state = S_1 if (np.random.uniform() < self.transitions[action][0]) else S_2

        # keep track of stay probability after first action
        if self.last_action is not None:    
            self.update_stay_prob(action)

        self.last_action = action
        self.last_is_common = self.is_common(action, self.state-1) 

        # return encoding of new state
        return self.encode_state()

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