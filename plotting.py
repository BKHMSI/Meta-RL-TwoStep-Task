import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm 
from tensorboard.backend.event_processing import event_accumulator

'''plotting functions'''

def plot_seeds(save_path, 
            load_path, 
            mode,
            y_lim=0.5,
            n_seeds=8, 
            base_seed=1111,
            title="Two-Step Task"):

    _, ax = plt.subplots()
    common_sum = np.array([0.,0.])
    uncommon_sum = np.array([0.,0.])

    for seed_idx in range(1, n_seeds + 1):

        ax.set_ylim([y_lim, 1.0])
        ax.set_ylabel('Stay Probability')
        
        path = os.path.join(load_path+f"_{seed_idx}", f"{os.path.basename(load_path)}_{seed_idx}_{base_seed*seed_idx}_{mode}.npy")
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

def print_rewards(save_path, load_path_mrl, load_path_emrl, title):

    mrl_cued   = np.load(os.path.join(load_path_mrl, "reward_cued.npy"))
    mrl_uncued = np.load(os.path.join(load_path_mrl, "reward_uncued.npy"))

    emrl_cued   = np.load(os.path.join(load_path_emrl, "reward_cued.npy"))
    emrl_uncued = np.load(os.path.join(load_path_emrl, "reward_uncued.npy"))

    print(f"MRL: Cued {mrl_cued.mean()} | Uncued {mrl_uncued.mean()}")
    print(f"EMRL: Cued {emrl_cued.mean()} | Uncued {emrl_uncued.mean()}")

def read_data(load_dir, tag="perf/avg_reward_10"):

    events = os.listdir(load_dir)
    for event in events:
        path = os.path.join(load_dir, event)
        ea = event_accumulator.EventAccumulator(path, size_guidance={ 
                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.AUDIO: 0,
                event_accumulator.SCALARS: 10_000,
                event_accumulator.HISTOGRAMS: 0,
        })
        
        ea.Reload()
        tags = ea.Tags()

        if tag not in tags["scalars"]: continue

        if len(ea.Scalars(tag)) == 10_000:
            return np.array([s.value for s in ea.Scalars(tag)])

    return None 

def plot_rewards_curve(save_path, load_path_epi, load_path_inc, n_seeds=10):

    epi_data = np.zeros((n_seeds, 10_000))
    inc_data = np.zeros((n_seeds, 10_000))

    for seed_idx in tqdm(range(n_seeds)):
        epi_event = read_data(load_dir=load_path_epi+f"_{seed_idx+1}")
        inc_event = read_data(load_dir=load_path_inc+f"_{seed_idx+1}")
        if epi_event is None or inc_event is None: 
            raise ValueError()
        epi_data[seed_idx] = epi_event 
        inc_data[seed_idx] = inc_event

    epi_mean = epi_data.mean(axis=0)
    inc_mean = inc_data.mean(axis=0)

    plt.plot(epi_mean)
    plt.plot(inc_mean)
    plt.legend(["Episodic", "Incremental"])
    plt.title("Episodic vs Incremental Training Curves")
    plt.savefig(save_path)


if __name__ == "__main__":

    ## Episodic Plot ##
    # plot_seeds(
    #     save_path="assets/ep_twostep_stayprob.png", 
    #     load_path="ckpt/TwoStepEp_12", 
    #     mode="episodic", 
    #     y_lim=0, 
    #     n_seeds=10, 
    #     title="Episodic"
    # )

    ## Compare Episodic vs Incremental Training Curves ##
    plot_rewards_curve(
        save_path="./assets/epi_inc_rewards.png",
        load_path_epi="./logs_ep/TwoStepEp_12",
        load_path_inc="./logs_ep/TwoStepEp_13",
        n_seeds=10
    )

