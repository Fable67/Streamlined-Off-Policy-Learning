import ptan
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

HID_SIZE = 64
ACTF = nn.ReLU
LOG_STD_MIN = -20
LOG_STD_MAX = 2


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size, hid_size, actf):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            actf(),
            nn.Linear(hid_size, hid_size),
            actf(),
            nn.Linear(hid_size, act_size),
        )

        self.apply(weights_init_)

    def forward(self, x):
        return self.mu(x)


class ModelTwinQ(nn.Module):
    def __init__(self, obs_size, act_size, hid_size):
        super(ModelTwinQ, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_size + act_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_size + act_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

        self.apply(weights_init_)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.q1(x), self.q2(x)


class Agent(ptan.agent.BaseAgent):
    def __init__(self, net, fixed_sigma_value, beta, device="cpu"):
        self.net = net
        self.device = device
        self.fixed_sigma_value = fixed_sigma_value
        self.beta = beta

    def get_actions(self, states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v = self.net(states_v)
        std_v = torch.zeros(mu_v.size()).to(self.device)
        std_v += self.fixed_sigma_value
        zeros_v = torch.zeros(mu_v.size()).to(self.device)
        normal = Normal(zeros_v, std_v)
        K_v = torch.tensor(mu_v.size()[1]).to(self.device)
        Gs_v = torch.sum(torch.abs(mu_v), dim=1).view(-1, 1)
        Gs_v = Gs_v / K_v
        # TODO: Check if additional devision by beta improves performance
        # Gs_v = Gs_v / self.beta
        ones_v = torch.ones(Gs_v.size()).to(self.device)
        Gs_mod1_v = torch.where(Gs_v >= 1, Gs_v, ones_v)
        mu_v = mu_v / Gs_mod1_v
        actions = torch.tanh(mu_v + normal.rsample())
        return actions

    def get_actions_deterministic(self, states_v):
        mu_v = self.net(states_v)
        K_v = torch.tensor(mu_v.size()[1]).to(self.device)
        Gs_v = torch.sum(torch.abs(mu_v), dim=1).view(-1, 1)
        Gs_v = Gs_v / K_v
        # TODO: Check if beta != 1 improves performance
        Gs_v = Gs_v / self.beta
        ones_v = torch.ones(Gs_v.size()).to(self.device)
        Gs_mod1_v = torch.where(Gs_v >= 1, Gs_v, ones_v)
        mu_v = mu_v / Gs_mod1_v
        actions = torch.tanh(mu_v)
        return actions

    def __call__(self, states, agent_states):
        actions = self.get_actions(states).data.cpu().numpy()
        return actions, agent_states


class RandomAgent(ptan.agent.BaseAgent):
    def __init__(self, act_size):
        self.act_size = act_size

    def __call__(self, states, agent_states):
        actions = np.random.rand(np.shape(states)[0], self.act_size) * 2 - 1
        return actions, agent_states
