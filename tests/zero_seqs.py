import sys
sys.path.append("..")

import numpy as np

from lib import infra

EPSILON = 1e-6


def action_reward_hook(our_state, bbox_state, rewards, next_states):
    action = np.random.randint(0, infra.n_actions, 1)[0]
    re_copy = np.copy(rewards)
#    print re_copy

    if our_state['last_reward'] is not None:
        n = np.linalg.norm(our_state['last_reward'] - rewards)
        if n < EPSILON:
            our_state['same_count'] += 1
        elif our_state['same_count'] > 0:
            print "Seq %d %s %s" % (our_state['same_count']+1, our_state['last_reward'].tolist(), re_copy.tolist())
            our_state['same_count'] = 0

    our_state['last_reward'] = re_copy
    for reward in rewards:
        our_state['rewards_set'].add(reward)
    return action


if __name__ == "__main__":
    np.random.seed(42)
    infra.setup_logging()
    infra.prepare_bbox()

    state = {
        'last_reward': None,
        'same_count': 1,
        'rewards_set': set()
    }

    infra.bbox_checkpoints_loop(state, action_reward_hook, verbose=False)
    print "Rewards set len: %d" % len(state['rewards_set'])
    print "Rewards set: %s" % (state['rewards_set'])
