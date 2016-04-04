import array, struct
import glob

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
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def append(self, state, action, reward, next_states):
        self.buffer.append((state, action, reward, next_states))
        while len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def __str__(self):
        return "ReplayBuffer: size={size}".format(size=len(self.buffer))

    # TODO: generate shuffle index
    # TODO: iterate over batches