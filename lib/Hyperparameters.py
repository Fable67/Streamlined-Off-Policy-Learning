import torch.nn as nn
from lib import common

ENV_ID = "RoboschoolHalfCheetah-v1"
GAMMA = 0.99
BATCH_SIZE = 256
LR_ACTOR = 0.0003
LR_CRITIC = 0.0003
REPLAY_SIZE = 1000000
REPLAY_INITIAL = 10000
TAU = 0.005
REWARD_STEPS = 1
STEPS_PER_EPOCH = 5000
ETA_INIT = 0.995
ETA_FINAL = 0.999
ETA_BASELINE_EPOCH = 100
ETA_AVG_SIZE = 20
C_MIN = 5000
FIXED_SIGMA_VALUE = 0.3
BETA_AGENT = 1
MAX_ITERATIONS = 1000000
HID_SIZE = 256
ACTF = nn.ReLU
OPTIMIZER = common.EmphasizingExperienceReplay
BETA_START = 0.5
BETA_END_ITER = 10000
ALPHA_PROB = 0.7
