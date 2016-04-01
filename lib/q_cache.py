def round_state(state, digits=2):
    return tuple(map(lambda v: round(v, digits), state))


class QCache:
    def __init__(self, gamma):
        self.gamma = gamma
        self.cache = set()
        self.reset()


    def reset(self):
        self.hits = 0
        self.misses = 0


    def transform_batch(self, states, qvals, next_states):
        for batch_idx, raw_state in enumerate(states):
            state = round_state(raw_state)

            for action, (qval, raw_next_state) in enumerate(zip(qvals[batch_idx], next_states[batch_idx])):
                next_state = round_state(raw_next_state)
                self.lookup(next_state)


    def state(self):
        return "[size={size}, hits={hits}, misses={misses}]".format(
                size=len(self.cache), hits=self.hits, misses=self.misses)


    def lookup(self, state):
        if state in self.cache:
            self.hits += 1
        else:
            self.misses += 1
            self.cache.add(state)
