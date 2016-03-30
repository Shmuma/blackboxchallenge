"""
Replay generator
"""
from time import time
from datetime import timedelta
import numpy as np
import argparse
import cPickle as pickle

from lib import infra, replays


def action_hook(our_state, bbox_state):
    action = np.random.randint(0, infra.n_actions, 1)[0]
    our_state['states'].append(bbox_state)
    our_state['actions'].append(action)
    return action


def reward_hook(our_state, reward, last_round):
    our_state['rewards'].append(reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", required=True, help="File name to save replay")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    infra.prepare_bbox()
    state = {
        'states': [],
        'actions': [],
        'rewards': [],
    }
    t = time()
    infra.bbox_loop(state, action_hook, reward_hook, verbose=False)
    replays.save_replay(zip(state['states'], state['actions'], state['rewards']), args.output)
    print("Replay of length {len} generated in {duration}".format(
            len=len(state['states']), duration=timedelta(seconds=time()-t)))