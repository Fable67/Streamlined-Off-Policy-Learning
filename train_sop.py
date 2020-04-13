import os
import ptan
import gym
import math
import time
import argparse
from tensorboardX import SummaryWriter
import numpy as np

from lib import model, common
from lib.Hyperparameters import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distrib
import torch.nn.functional as F


try:
    import roboschool
except:
    print("A problem occured when trying to import roboschool. Maybe not installed?")

try:
    import pybullet_envs
except:
    print("A problem occured when trying to import pybullet_envs. Maybe not installed?")


@torch.no_grad()
def test_net(agent, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            states_v = ptan.agent.float32_preprocessor([obs])
            states_v = states_v.to(device)
            mu_v = agent.get_actions_deterministic(states_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-e", "--env", default=ENV_ID,
                        help="Environment id, default=" + ENV_ID)
    parser.add_argument("-i", "--iterations", type=int, default=MAX_ITERATIONS,
                        help="Maximum number of iterations, default=" + str(MAX_ITERATIONS))
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "sop-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(args.env)
    test_env = gym.make(args.env)

    act_net = model.ModelActor(
        env.observation_space.shape[0],
        env.action_space.shape[0], HID_SIZE, ACTF).to(device)
    twinq_net = model.ModelTwinQ(
        env.observation_space.shape[0],
        env.action_space.shape[0], HID_SIZE).to(device)
    print(act_net)
    print(twinq_net)

    tgt_twinq_net = ptan.agent.TargetNet(twinq_net)

    writer = SummaryWriter(comment="-sop_" + args.name)
    agent = model.Agent(act_net, FIXED_SIGMA_VALUE, BETA, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    buffer = common.EmphasizingExperienceReplay(
        exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LR_ACTOR)
    q1_opt = optim.Adam(twinq_net.q1.parameters(), lr=LR_CRITIC)
    q2_opt = optim.Adam(twinq_net.q2.parameters(), lr=LR_CRITIC)

    frame_idx = 0
    epoch_step = 0
    recent_total_episode_return = 0
    recent_n_episodes = 0
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(
                writer, batch_size=10) as tb_tracker:
            while True:
                rewards_steps = []
                while len(rewards_steps) == 0:
                    buffer.populate(1)
                    rewards_steps = exp_source.pop_rewards_steps()
                rewards, steps = zip(*rewards_steps)
                ep_ret = rewards[0]
                ep_len = steps[0]
                frame_idx += ep_len
                epoch_step += ep_len
                recent_total_episode_return += ep_ret
                recent_n_episodes += 1
                tb_tracker.track("episode_steps", ep_len, frame_idx)
                tracker.reward(ep_ret, frame_idx)
                if len(buffer) < REPLAY_INITIAL:
                    continue

                ref_vals = 0
                q1_loss = 0
                q2_loss = 0
                q = 0
                actor_loss = 0

                eta_current = buffer.get_eta(ETA_INIT, ETA_FINAL, ETA_BASELINE_EPOCH, ETA_AVG_SIZE)
                ck_list = buffer.get_cks(ep_len, eta_current)

                for k in range(ep_len):
                    c_k = ck_list[k]
                    if c_k < C_MIN:
                        c_k = C_MIN

                    batch = buffer.sample(c_k, BATCH_SIZE)
                    states_v, actions_v, ref_vals_v = \
                        common.unpack_batch(batch, tgt_twinq_net.target_model,
                                            agent, GAMMA ** REWARD_STEPS, device)

                    with torch.no_grad():
                        ref_vals += ref_vals_v.mean()

                    # TODO: Check if seperate optimizers for the q functions improve performance
                    # TwinQ
                    q1_v, q2_v = twinq_net(states_v, actions_v)
                    q1_loss_v = (q1_v.squeeze() - ref_vals_v.detach()).pow(2).mean()
                    q2_loss_v = (q2_v.squeeze() - ref_vals_v.detach()).pow(2).mean()
                    with torch.no_grad():
                        q1_loss += q1_loss_v
                        q2_loss += q2_loss_v

                    # Actor
                    acts_v = agent.get_actions_deterministic(states_v)
                    q_v = torch.min(*twinq_net(states_v, acts_v))
                    act_loss_v = (- q_v.squeeze()).mean()

                    with torch.no_grad():
                        q += q_v.mean()
                        actor_loss += act_loss_v

                    q1_opt.zero_grad()
                    q1_loss_v.backward()
                    q1_opt.step()

                    q2_opt.zero_grad()
                    q2_loss_v.backward()
                    q2_opt.step()

                    act_opt.zero_grad()
                    act_loss_v.backward()
                    act_opt.step()

                    tgt_twinq_net.alpha_sync(alpha=1. - TAU)

                tb_tracker.track("ref_vals", ref_vals / ep_len, frame_idx)
                tb_tracker.track("q1_loss", q1_loss / ep_len, frame_idx)
                tb_tracker.track("q2_loss", q2_loss / ep_len, frame_idx)
                tb_tracker.track("q", q / ep_len, frame_idx)
                tb_tracker.track("actor_loss", actor_loss / ep_len, frame_idx)

                if epoch_step > STEPS_PER_EPOCH:
                    epoch_step -= STEPS_PER_EPOCH
                    ts = time.time()
                    rewards, steps = test_net(agent, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)

                    buffer.store_epoch_performance(recent_total_episode_return / recent_n_episodes)
                    recent_total_episode_return = 0
                    recent_n_episodes = 0

                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards

                if frame_idx > args.iterations:
                    break
