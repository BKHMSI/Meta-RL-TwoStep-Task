import os
import numpy as np
import matplotlib.pyplot as plt


'''helpers'''

def plot_seeds(save_path, 
            load_path, 
            n_seeds=8, 
            base_seed=1111,
            title="Two-Step Task"):

    _, ax = plt.subplots()
    common_sum = np.array([0.,0.])
    uncommon_sum = np.array([0.,0.])

    for seed_idx in range(1, n_seeds + 1):

        ax.set_ylim([0.5, 1.0])
        ax.set_ylabel('Stay Probability')
        
        path = os.path.join(load_path+f"_{seed_idx}", f"TwoStep_60_{seed_idx}_{base_seed*seed_idx}.npy")
        stay_probs = np.load(path)

        common = [stay_probs[0,0,0], stay_probs[1,0,0]]
        uncommon = [stay_probs[0,1,0], stay_probs[1,1,0]]
        
        common_sum += np.array(common)
        uncommon_sum += np.array(uncommon)

        ax.set_xticks([1.5,3.5])
        ax.set_xticklabels(['Rewarded', 'Unrewarded'])

        plt.plot([1,3], common, 'o', color='black')
        plt.plot([2,4], uncommon, 'o', color='black')
        
    c  = plt.bar([1,3],  (1. / n_seeds) * common_sum, color='b', width=0.5)
    uc = plt.bar([2,4], (1. / n_seeds) * uncommon_sum, color='r', width=0.5)
    ax.legend( (c[0], uc[0]), ('Common', 'Uncommon') )
    ax.set_title(title)
    plt.savefig(save_path)


if __name__ == "__main__":
    load_path = "./ckpt/TwoStep_60"
    save_path = "./assets/twostep_stayprob.png"

    plot_seeds(save_path, load_path)
