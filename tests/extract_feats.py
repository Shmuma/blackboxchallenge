import sys
sys.path.append("..")
import os

import numpy as np

from lib import infra

EPSILON = 1e-6


def action_reward_hook(our_state, bbox_state, rewards, next_states):
    action = np.random.randint(0, infra.n_actions, 1)[0]

    for act in range(infra.n_actions):
        reward = rewards[act]
        for idx in range(infra.n_features):
            our_state['fds'][idx].write("%f,%f\n" % (reward, next_states[act][idx]))

    return action


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    infra.setup_logging()
    infra.prepare_bbox()
    out_path = "out/seed=%d" % SEED
    os.makedirs(out_path)

    state = {
        'fds': [open(out_path + "/%02d-reward-state.txt" % feat, "w+") for feat in range(infra.n_features)]
    }

    infra.bbox_checkpoints_loop(state, action_reward_hook, verbose=1000)

    for fd in state['fds']:
        fd.close()