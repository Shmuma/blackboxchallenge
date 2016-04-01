import numpy as np

def round_state(state, digits=2):
    return tuple(map(lambda v: round(v, digits), state))


class QCache:
    def __init__(self, gamma):
        self.gamma = gamma
        self.cache = dict()
        self.reset()


    def reset(self):
        self.hits = 0
        self.misses = 0


    def transform_batch(self, states, qvals, next_states):
        for batch_idx, raw_state in enumerate(states):
            new_q = np.zeros_like(qvals[batch_idx])
            conv_next_states = []
            for action, (qval, raw_next_state) in enumerate(zip(qvals[batch_idx], next_states[batch_idx])):
                next_state = round_state(raw_next_state)
                conv_next_states.append(next_state)
                next_q = self.lookup(next_state)
                new_q[action] = qvals[batch_idx][action]
                if next_q is not None:
                    new_q[action] += self.gamma * next_q
            for action, next_state in enumerate(conv_next_states):
                self.save_cache(next_state, new_q[action])
            qvals[batch_idx] = new_q


    def state(self):
        return "[size={size}, hits={hits}, misses={misses}]".format(
                size=len(self.cache), hits=self.hits, misses=self.misses)


    def lookup(self, state):
        res = self.cache.get(state)
        if res is not None:
            self.hits += 1
        else:
            self.misses += 1
        return res


    def save_cache(self, state, qval):
        self.cache[state] = qval