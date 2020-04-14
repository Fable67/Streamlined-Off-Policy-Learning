import argparse
import gym

from lib import model
from lib.Hyperparameters import *

import numpy as np
import torch

import torch.nn as nn


try:
    import roboschool
except:
    print("A problem occured when trying to import roboschool. Maybe not installed?")

try:
    import pybullet_envs
except:
    print("A problem occured when trying to import pybullet_envs. Maybe not installed?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID,
                        help="Environment name to use, default=" + ENV_ID)
    parser.add_argument(
        "-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    parser.add_argument("--eval", default=False, action='store_true', help='Evaluates Agent')
    args = parser.parse_args()

    reward_eval_env = gym.make(args.env)

    env = gym.make(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record, force=True)

    net = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0],
                           HID_SIZE, ACTF)
    net.load_state_dict(torch.load(args.model))
    agent = model.Agent(net, FIXED_SIGMA_VALUE, BETA)

    if args.eval:
        print("Evaluating Agent...")
        rewards = 0.0
        steps = 0
        for _ in range(100):
            obs = reward_eval_env.reset()
            while True:
                obs_v = torch.FloatTensor([obs])
                mu_v = agent.get_actions_deterministic(obs_v)
                action = mu_v.squeeze(dim=0).data.cpu().numpy()
                obs, reward, done, _ = reward_eval_env.step(action)
                rewards += reward
                steps += 1
                if done:
                    break
        print("The Agent was able to reach an average reward of %.3f over 100 consecutive episodes" %
              (rewards / 100))

    obs = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        mu_v = agent.get_actions_deterministic(obs_v)
        action = mu_v.squeeze(dim=0).data.cpu().numpy()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
        if args.record is None:
            env.render()
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))

    if args.record is None:
        env.close()
