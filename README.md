# Streamlined Off-Policy Learning
<br/>

## Description
------------
Implementation of [Striving for Simplicity and Performance in Off-Policy DRL: Output Normalization and Non-Uniform Sampling](https://arxiv.org/abs/1910.02208) using the PyTorch Deep Learning Framework and PyTorch Agent Net (PTAN) Reinforcement Learning Toolkit. <br>
The specific algorithm implemented in this repository is the following:
<img src="./images/SOP+ERE.png">  
There is also the option to use an optmizer I created myself that is kind of a hybrid between the one above and Prioritized Experience Replay. I haven't tested it much and it 
definitely needs further investigation, however, in case someone wants to play around with it, you can!!!
<br/>

## Requirements
------------
*   [mujoco-py](https://github.com/openai/mujoco-py)
*   [TensorboardX](https://github.com/lanpa/tensorboardX)
*   [PyTorch](http://pytorch.org/)
*   [PTAN](https://github.com/Shmuma/ptan)
<br/>

## Usage
------------
```
train_sop.py [-h] [--cuda] [--name NAME] [--env ENV] [--iterations ITERATIONS]

play_sop.py [-h] [--eval] [--model MODEL] [--env ENV] [--record RECORD]
```
All Hyperparameters can be changed in the file /lib/Hyperparameters.py.
<br/>


## Hyperparameters
------------

*   ENV_ID = "RoboschoolHalfCheetah-v1"                  (Name of the environment.)
*   GAMMA = 0.99                                         (Discount factor.)
*   BATCH_SIZE = 256                                     (Batch size for training.)
*   LR_ACTOR = 0.0003                                    (Learning rate of the Actor/policy.)
*   LR_CRITIC = 0.0003                                   (Learning rate of the Critics/Q-Networks.)
*   REPLAY_SIZE = 1000000                                (Maximum size of the replay buffer.)
*   REPLAY_INITIAL = 10000                               (Minimum size of the replay buffer to begin training.)
*   TAU = 0.005                                          (Target smoothing coefficient.)
*   REWARD_STEPS = 1                                     (Number of rollouts for Q approximation.)
*   STEPS_PER_EPOCH = 5000                               (Number of steps an epoch has.)
*   ETA_INIT = 0.995                                     (Initial eta for recent experience sampling.)
*   ETA_FINAL = 0.999                                    (Final eta for recent experience sampling.)
*   ETA_BASELINE_EPOCH = 100                             (Minimum number of epochs to approximate baseline for improvement normalization with.)
*   ETA_AVG_SIZE = 20                                    (Number of epochs to average performance with.)
*   C_MIN = 5000                                         (Minimum number of recent samples.)
*   FIXED_SIGMA_VALUE = 0.29                             (Sigma for additive-gaussian-noise in actors action selection.)
*   BETA_AGENT = 1                                       (Beta for regularization in action normalization process.)
*   MAX_ITERATIONS = 3000000                             (Maximum number of iterations.)
*   HID_SIZE = 256                                       (Number of Neurons in Actors and Critics Hidden Layers.)
*   ACTF = nn.ReLU                                       (Activation function used in Actor Network.)
*   BUFFER = common.EmphasizingExperienceReplay          (Type of Replay Buffer used.)
*   BETA_START = 0.4                                     (Starting value for beta for prioritized experience.)
*   BETA_END_ITER = 10000                                (At which iteration beta for prioritized experience should be 1.)
*   ALPHA_PROB = 0.6                                     (Exponent for probabilities in prioritized experience.)
*	MUNCHAUSEN = False									 (Use Munchausen)
<br/>


## Results
### HalfCheetah-v3
<img src="./videos/SOP-HalfCheetah-v3.gif">
<br/>

### Humanoid-v3
<img src="./videos/SOP-Humanoid-v3.gif">
<br/>

### LunarLanderContinuous-v2
<img src="./videos/SOP-LunarLanderContinuous-v2.gif">
<br/>


## Mentions
------------
This implementation is adapted from [Shmumas](https://github.com/Shmuma) [SAC implementation using PTAN](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/sac-experiment/Chapter19/06_train_sac.py). <br/>
It is also influenced by the [official implementation](https://github.com/AutumnWu/Streamlined-Off-Policy-Learning).
<br/>


## Reference
```shell
@misc{wang2019striving,
    title={Striving for Simplicity and Performance in Off-Policy DRL: Output Normalization and Non-Uniform Sampling},
    author={Che Wang and Yanqiu Wu and Quan Vuong and Keith Ross},
    year={2019},
    eprint={1910.02208},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
