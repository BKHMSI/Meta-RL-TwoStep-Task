﻿# Meta-RL: Episodic/Contextual and Incremental Two-Step Task (PyTorch)

In this repository, I reproduce the results of [Prefrontal Cortex as a Meta-Reinforcement Learning System](https://www.nature.com/articles/s41593-018-0147-8)<sup>1</sup>, [Episodic Control as Meta-Reinforcement Learning](https://www.biorxiv.org/content/10.1101/360537v2)<sup>2</sup> and [Been There, Done That: Meta-Learning with Episodic Recall](https://arxiv.org/abs/1805.09692)<sup>3</sup> on variants of the sequential decision making "Two Step" task originally introduced in [Model-based Influences on Humans’ Choices and Striatal Prediction Errors](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3077926/)<sup>4</sup>. You will find below a description of the task with results, along with a brief overview of Meta-RL and its connection to neuroscience, as well as details covering the structure of the code.

<table align="center">
    <tr>
        <th>Episodic Two-Step Task</th>
        <th>Episodic LSTM</th>
    </tr>
    <tr>
        <td align="center" width="50%"><img alt="Two-Step Task Illustration" src="assets/episodic_task.png"></td>
        <td align="center" width="50%"><img alt="Episodic LSTM" src="assets/episodic_lstm.png"></td>
    </tr>
</table>

*Note: I have a related repository on the ["Harlow" visual fixation task](https://github.com/BKHMSI/Meta-RL-Harlow) in case you are interested :)

**I aim to write a blog post to accompany this repository, so stay tuned!

## Overview of Meta-RL

In recent years, deep reinforcement learning (deep-RL) have been in the forefront of artificial intelligence research since DeepMind's seminal work on DQN <sup>5</sup> that was able to solve a wide range of Atari games by just looking at the raw pixels, as a human would. However, there remained a major issue that disqualified it as a plausible model of human learning, and that is the sample efficiency problem. It basically refers "to the amount of data required for a learning system to attain any chosen target level of performance" <sup>6</sup>. In other words, a task that would take a biological brain a matter of minutes to master would require many orders of magnitude more training data for a deep-RL artificial agent. Botvinick et al. (2019) <sup>6</sup> identify two main sources of slowness in deep-RL: the need for *incremental parameter adjustment* and starting with a *weak inductive bias*. I will be going into more details of each in my blog post. However, they note that subsequent research has shown that it is possible to train artificial agents in a sample efficient manner, and that is by (1) augmenting it with an episodic memory system to avoid redundant exploration and leverage prior experience more effectively, and (2) using a meta-learning approach by training the agent on a series of structurally interrelated tasks that can strengthen the inductive bias (narrowing the hypothesis set) which enables the agent to hone-in to a valid solution much faster <sup>6</sup>.

### Connection with Neuroscience

DeepMind have been preaching about the importance of neuroscience and artificial intelligence research to work together in what they call a virtuous circle; where each field will inspire and drive forward the other <sup>7</sup>. I must admit that they were the reason I joined an MSc program in Computational Cognitive Neuroscience after working on AI for a couple of years now. In short, they were indeed able to show that meta-RL - which was drawn from the machine learning literature - is able to explain a wide range of neuroscientific findings as well as resolve many of the prevailing quandaries in reward-based learning <sup>1</sup>. They do so by conceptualizing the prefrontal cortex along with its subcortical components (basal ganglia and thalamic nuclei) as its own free standing meta-RL system. Concretely, they show that dopamine driven synaptic plasticity, that is model free, gives rise to a second more efficient model-based RL algorithm implemented in the prefrontal's network activation dynamics <sup>1</sup>. In this repository, I reproduce one of their simulations (the two-step task) that showcases the emergence of model-based learning in accord with observed behavior in both humans and rodents <sup>1</sup>. 

The episodic meta-RL variant was proposed by Ritter et al. (2018) <sup>2, 3</sup> and is partly inspired by evidence that episodic memory retrieval in humans operate through reinstatement that recreates patterns of activity in neural circuits supporting working memory <sup>2</sup>. This presents a new theory in which human decision making can be seen as an interplay between working and episodic memory, that is itself learned through training to maximise rewards on a distribution of tasks. The episodic memory system is implemented as a differentiable neural dictionary <sup>8</sup> that stores the task context as keys and the LSTM cell states as values. This will be expanded upon as well in the accompanied blog post. 

## The Two-Step Task

This task has been widely used in the neuroscience literature to distinguish the contribution of different systems viewed to support decision making. The variant I am using here was developed to disassociate a model-free system, that caches values of actions in states, from a model-based system that learns an internal model of the environment and evaluates the values of actions through look-ahead planning <sup>9</sup>. The purpose here is to see whether the model-free algorithm used to train the weights (which is A2C<sup>10</sup> in this case) gives rise to behavior emulating model-based strategy.

## Results

The results below, which this code reproduces, shows the stay probability as a function of the previous transition type (common or uncommon) and whether the agent was rewarded in the last trial or not. This probability basically reflects the selection of the same action as in the previous trial. 

What we are looking for is a higher stay probability if the previous trial used a common transition and was rewarded or the opposite if it was not rewarded. This is indeed what we see below, which implies model-based behavior. I strongly encourage to check the referenced papers for a more detailed description of the task and hyperparameters used in those simulations. 

<table align="center">
    <tr>
        <th>Published Result <sup>1</sup></th>
        <th>My Result</th>
    </tr>
    <tr>
        <td align="center" width="50%"><img alt="Stay Prob Two-Step Task Published" src="assets/pub_twostep_stayprob.png"></td>
        <td align="center" width="50%"><img alt="Stay Prob Two-Step Task" src="assets/twostep_stayprob.png"></td>
    </tr>
</table>

### Episodic vs Incremental 

The incremental version uses the exact same setup and model as the episodic one with the same hyperparameters but with the reinstatement gate set to 0, therefore no memories are retrieved. 

<table align="center" width="100%">
    <tr>
        <th colspan="3">Published Result <sup>2</sup></th>
        <th colspan="3">My Result</th>
    </tr>
    <tr>
        <th> Incremental Uncued </th>
        <th> Incremental Cued </th>
        <th> Episodic </th>
        <th> Incremental Uncued </th>
        <th> Incremental Cued </th>
        <th> Episodic </th>
    </tr>
    <tr>
        <td align="center" width="16.6%">
            <img alt="Published Incremental Uncued Stay Prob Result" src="assets/pub_inc_uc_twostep_stayprob.png">
        </td>
        <td align="center" width="16.6%">
            <img alt="Published Incremental Cued Stay Prob Result" src="assets/pub_inc_cued_twostep_stayprob.png">
        </td>
        <td align="center" width="16.6%">
            <img alt="Published Episodic Stay Prob Result" src="assets/pub_ep_twostep_stayprob.png">
        </td>
        <td align="center" width="16.6%">
            <img alt="My Incremental Uncued Stay Prob Result" src="assets/inc_uc_twostep_stayprob.png">
        </td>
        <td align="center" width="16.6%">
            <img alt="My Incremental Cued Stay Prob Result" src="assets/inc_cued_twostep_stayprob.png">
        </td>
        <td align="center" width="16.6%">
            <img alt="My Episodic Stay Prob Result" src="assets/ep_twostep_stayprob.png">
        </td>
    </tr>
</table>

### Training Curves 

Those are the training trajectories of the episodic and incremental variants (each of which consists of 10 runs with different random seeds). We can see that the episodic version on average converges faster and accumulates more rewards early on as it makes better use of prior experience using the long-term memory. 

<table align="center">
    <tr>
        <th>Episodic Training Curve</th>
        <th>Incremental Training Curve</th>
    </tr>
    <tr>
        <td align="center" width="50%"><img alt="Episodic Training Curve" src="assets/episodic_rewards_training.png"></td>
        <td align="center" width="50%"><img alt="Incremental Training Curve" src="assets/incremental_rewards_training.png"></td>
    </tr>
</table>


## Code Structure

``` bash
Meta-RL-TwoStep-Task
├── LICENSE
├── README.md
├── episodic.py # reproduces results of [2] and [3]
├── vanilla.py  # reproduces results of [1]
├── plotting.py # plots extra graphs
└── configs
    ├── ep_two_step.yaml # configuration file with hyperparameters for episodic.py
    └── two_step.yaml # configuration file with hyperparamters for vanilla.py
└── tasks
    ├── ep_two_step.py # episodic two-step task
    └── two_step.py # vanilla two-step task
└── models
    ├── a2c_lstm.py # advantage actor-critic (a2c) algorithm with working memory
    ├── a2c_dnd_lstm.py # a2c algorithm with working memory and long-term (episodic) memory
    ├── dnd.py # episodic memory as a differentiable neural dictionary
    ├── ep_lstm.py # episodic lstm module wrapper
    └── ep_lstm_cell.py # episodic lstm cell with extra reinstatement gate
```

## References

1. Wang, J., Kurth-Nelson, Z., Kumaran, D., Tirumala, D., Soyer, H., Leibo, J., Hassabis, D., & Botvinick, M. (2018). [Prefrontal Cortex as a Meta-Reinforcement Learning System](https://www.nature.com/articles/s41593-018-0147-8). *Nat Neurosci*, **21**, 860–868.

2. Ritter, S., Wang, J., Kurth-Nelson, Z., & Botvinick, M. (2018). [Episodic Control as Meta-Reinforcement Learning](https://www.biorxiv.org/content/10.1101/360537v2). *bioRxiv*.

3. Ritter, S., Wang, X., Kurth-Nelson, Z., Jayakumar, M., Blundell, C., Pascanu, R., & Botvinick, M. (2018). [Been There, Done That: Meta-Learning with Episodic Recall](https://arxiv.org/abs/1805.09692). *ICML*, 4351–4360. 

4. Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P., & Dolan, R. J. (2011).[Model-based Influences on Humans’ Choices and Striatal Prediction Errors](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3077926/). *Neuron*, 69(6), 1204–1215. https://doi.org/10.1016/j.neuron.2011.02.027

5. Mnih, Volodymyr & Kavukcuoglu, Koray & Silver, David & Rusu, Andrei & Veness, Joel & Bellemare, Marc & Graves, Alex & Riedmiller, Martin & Fidjeland, Andreas & Ostrovski, Georg & Petersen, Stig & Beattie, Charles & Sadik, Amir & Antonoglou, Ioannis & King, Helen & Kumaran, Dharshan & Wierstra, Daan & Legg, Shane & Hassabis, Demis. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). Nature. 518. 529-33. 10.1038/nature14236. 

6. Botvinick, Mathew & Ritter, Sam & Wang, Jane & Kurth-Nelson, Zeb & Blundell, Charles & Hassabis, Demis. (2019). [Reinforcement Learning, Fast and Slow](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(19)30061-0). Trends in Cognitive Sciences. 23. 10.1016/j.tics.2019.02.006. 

7. Hassabis, Demis & Kumaran, Dharshan & Summerfield, Christopher & Botvinick, Matthew. (2017). [Neuroscience-Inspired Artificial Intelligence](https://www.cell.com/neuron/fulltext/S0896-6273(17)30509-3). Neuron. 95. 245-258. 10.1016/j.neuron.2017.06.011. 

8. Pritzel, Alexander & Uria, Benigno & Srinivasan, Sriram & Puigdomènech, Adrià & Vinyals, Oriol & Hassabis, Demis & Wierstra, Daan & Blundell, Charles. (2017). [Neural Episodic Control](https://arxiv.org/abs/1703.01988).

9. Jane X. Wang and Zeb Kurth-Nelson and Dhruva Tirumala and Hubert Soyer and Joel Z. Leibo and Rémi Munos and Charles Blundell and Dharshan Kumaran and Matthew Botvinick (2016). [Learning to reinforcement learn](https://arxiv.org/abs/1611.05763). CoRR, abs/1611.05763.

10. Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D. & Kavukcuoglu, K.. (2016). [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783). Proceedings of The 33rd International Conference on Machine Learning, in *PMLR* 48:1928-1937

## Acknowledgments

I would like to give a shout out to those repositories and blog posts. They were of great help to me when implementing this project. Make sure to check them out!

- https://github.com/qihongl/dnd-lstm 
- https://github.com/mtrazzi/two-step-task
- https://github.com/lnpalmer/A2C 
- https://github.com/rpatrik96/pytorch-a2c
- https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html 
