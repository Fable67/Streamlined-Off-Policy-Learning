import numpy as np
import torch
import torch.distributions as distr

import ptan


@torch.no_grad()
def unpack_batch(batch, tgt_twinq_net, agent, last_val_gamma: float, device="cpu"):

    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)

    # handle rewards
    rewards = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_acts_v = agent.get_actions(last_states)
        last_q_v = torch.min(*tgt_twinq_net(last_states_v, last_acts_v))
        last_q = last_q_v.squeeze().data.cpu().numpy()

        rewards[not_done_idx] += last_val_gamma * last_q

    ref_q_v = torch.FloatTensor(rewards).to(device)
    return states_v, actions_v, ref_q_v


class EmphasizingExperienceReplay:
    def __init__(self, experience_source, buffer_size, max_epochs=5000):
        assert isinstance(experience_source, (ptan.experience.ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        assert isinstance(max_epochs, int)
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0
        self.epoch_performance_buf = np.zeros(max_epochs, dtype=np.float32)
        self.current_epoch = 0
        self.max_improvement = 1e-5

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, priority_count, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer

        max_index = min(int(priority_count), len(self.buffer))
        recent_relative_idxs = -np.random.randint(0, max_index, size=batch_size)
        recent_idxs = (self.pos - 1 + recent_relative_idxs) % len(self.buffer)
        return [self.buffer[idx] for idx in recent_idxs]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)

    def store_epoch_performance(self, performance):
        self.epoch_performance_buf[self.current_epoch] = performance
        self.current_epoch += 1

    def get_eta(self, eta_init=0.994, eta_final=0.999, baseline_epoch=100, avg_size=20):
        if self.current_epoch < baseline_epoch:
            return eta_init

        current_performance = self.epoch_performance_buf[self.current_epoch
                                                         - avg_size:self.current_epoch].mean()
        previous_performance = self.epoch_performance_buf[self.current_epoch -
                                                          100:self.current_epoch - 100 + avg_size].mean()
        recent_improvement = current_performance - previous_performance
        if recent_improvement > self.max_improvement:
            self.max_improvement = recent_improvement
        interpolation = recent_improvement / self.max_improvement
        interpolation = np.clip(interpolation, a_min=0, a_max=1)
        auto_eta = eta_init * interpolation + eta_final * (1 - interpolation)
        return auto_eta

    def get_cks(self, num_updates, eta_current):
        ck_list = np.zeros(num_updates, dtype=int)
        for k in range(num_updates):  # compute ck for each k, using formula for old data first update
            ck_list[k] = int(self.capacity * eta_current ** (k * 1000 / num_updates))
        return ck_list
