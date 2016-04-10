"""
Replay generator
"""
from time import time
from datetime import timedelta
import numpy as np
import argparse

from lib import infra, replays


def action_hook(our_state, bbox_state):
#    if np.random.random() < our_state['alpha']:
    action = np.random.randint(0, infra.n_actions, 1)[0]

    our_state['state'].append(bbox_state)
    our_state['state'] = our_state['state'][-our_state['history']:]
    our_state['action'] = action
    return action


def reward_hook(our_state, reward, last_round):
    if not last_round and len(our_state['state']) == our_state['history']:
        next_state = our_state['state'][-(our_state['history']-1):]
        next_state.append(infra.bbox.get_state())
        our_state['writer'].append_small(our_state['state'], our_state['action'],
                                         reward, next_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=4, help="States to track, default=4")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", required=True, help="File name to save replay")
    parser.add_argument("--alpha", type=float, default=0.0, help="Fraction of random steps, by default=1.0")
    parser.add_argument("--steps", type=int, help="Limit of simulator steps. Default = no limit")
    parser.add_argument("--bbox", default=None, help="BBox implementation to use, default=contest")
    args = parser.parse_args()
    print args
    infra.init(args.bbox)
    if args.seed is not None:
        np.random.seed(args.seed)

    infra.setup_logging()
    infra.prepare_bbox()
    state = {
        'history': args.history,
        'alpha': args.alpha,
        'writer': replays.ReplayWriter(args.output),
        'state': [],
    }
    try:
        t = time()
        infra.bbox_loop(state, action_hook, reward_hook, verbose=1, max_steps=args.steps)

        print("Replay generated in {duration}".format(
                duration=timedelta(seconds=time()-t)))
    finally:
        state['writer'].close()