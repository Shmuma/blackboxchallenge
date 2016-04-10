import numpy as np

class GridWorldState:
    def __init__(self, size, start, end, rewards):
        self.size = size
        self.start = start
        self.end = end
        self.init_rewards = rewards
        self.checkpoints = []

        self.reset()
        self.rewards = {}
        self.pos = start
        self.score = 0.0
        self.time = 0

    def reset(self):
        self.rewards = {(x, y): reward for x, y, reward in self.init_rewards}
        self.pos = self.start
        self.score = 0.0
        self.time = 0

    def __str__(self):
        return "Grid: pos={pos}, score={score}, time={time}, rewards={rewards}".format(
            pos=self.pos, score=self.score, time=self.time, rewards=self.rewards
        )

    def state(self):
        st = np.zeros((self.num_features(), ))
        st[0] = self.pos[0]
        st[1] = self.pos[1]
        return st

    def num_features(self):
        return 36

    def num_actions(self):
        return 4

    def get_new_pos(self, action):
        deltas = [(-1, 0), (0, 1), (0, -1), (1, 0)]
        x, y = self.pos
        x += deltas[action][0]
        y += deltas[action][1]
        if x < 0 or y < 0:
            return self.pos
        if x >= self.size[0] or y >= self.size[1]:
            return self.pos
        return x, y

    def do_action(self, action):
        new_pos = self.get_new_pos(action)
        r = self.rewards.pop(new_pos, 0.0)
        self.score += r
        self.time += 1
        self.pos = new_pos
        return new_pos != self.end

    def create_checkpoint(self):
        self.checkpoints.append((self.pos, self.score, self.time, dict(self.rewards)))
        return len(self.checkpoints)

    def clear_all_checkpoints(self):
        self.checkpoints = []

    def load_from_checkpoint(self, chp_id):
        self.pos, self.score, self.time, rewards = self.checkpoints[chp_id-1]
        self.rewards = dict(rewards)

    def action_name(self, action):
        return ["l", "u", "d", "r"][action]

