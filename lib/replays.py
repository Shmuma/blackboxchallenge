import array, struct
import glob

import numpy as np

import features

def pack_item(item):
    bbox_state, action, reward = item
    return array.array('f', bbox_state.tolist()).tostring(), action, reward


def pack_state(state):
    return array.array('f', state.tolist())


class ReplayWriter:
    def __init__(self, file_prefix):
        self.states_fd = open(file_prefix + ".states", "w+")
        self.actions_fd = open(file_prefix + ".actions", "w+")
        self.rewards_fd = open(file_prefix + ".rewards", "w+")
        self.next_states_fd = open(file_prefix + ".next_states", "w+")

    def close(self):
        self.states_fd.close()
        self.actions_fd.close()
        self.rewards_fd.close()
        self.next_states_fd.close()

    def append(self, state, action, rewards, next_states):
        pack_state(state).tofile(self.states_fd)
        self.actions_fd.write(struct.pack('b', action))
        self.rewards_fd.write(struct.pack('ffff', *rewards))
        for next_state in next_states:
            pack_state(next_state).tofile(self.next_states_fd)

    def append_small(self, states, action, reward, next_states):
        for s in states:
            pack_state(s).tofile(self.states_fd)
        self.actions_fd.write(struct.pack('b', action))
        self.rewards_fd.write(struct.pack('f', reward))
        for s in next_states:
            pack_state(s).tofile(self.next_states_fd)


def discover_replays(path):
    res = []
    suffix = ".states"
    for p in glob.glob(path + "/*" + suffix):
        res.append(p[:-len(suffix)])
    return res


class ReplayBuffer:
    def __init__(self, capacity, batch):
        self.capacity = capacity
        self.batch = batch
        self.buffer = []
        self.epoches = 0

    def append(self, state, rewards, next_states):
        # if state_history == 1, we don't need to do anything
        st = np.copy(state)

        if st.shape[0] == 1:
            next_st = np.copy([next_states])
        else:
            # combine state history and next_4_state into full next states history
            top_cur_state = st[:-1, :]
            v = []
            for next_state in next_states:
                v.append([np.vstack([next_state, top_cur_state])])
            next_st = np.concatenate(v)

        # transform features
        st = np.apply_along_axis(features.transform, 1, st)
        next_st = np.apply_along_axis(features.transform, 2, next_st)

        self.buffer.append((st, np.copy(rewards), next_st))

    def reshuffle(self):
        """
        Regenerate shuffled batch. Should be called after batch of appends.
        """
        while len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        self.shuffle = np.random.permutation(len(self.buffer))
        self.batch_idx = 0

    def next_batch(self):
        """
        Return next batch of data
        :return:
        """
        if (self.batch_idx + 1) * self.batch > len(self.buffer):
            self.reshuffle()
            self.epoches += 1

        states = []
        rewards = []
        next_states = []

        for idx in self.shuffle[self.batch_idx*self.batch:(self.batch_idx+1)*self.batch]:
            state, reward, next_4_state = self.buffer[idx]
            states.append(state)
            rewards.append(reward)
            next_states.append(next_4_state)

        self.batch_idx += 1
        return states, rewards, next_states

    def __str__(self):
        return "ReplayBuffer: size={size}, batch={batch}, epoch={epoch}".format(
                size=len(self.buffer), batch=self.batch_idx, epoch=self.epoches)
