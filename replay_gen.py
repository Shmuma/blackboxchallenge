"""
Replay generator
"""
from time import time
from datetime import timedelta
import numpy as np
import argparse

from lib import infra, replays


def action_reward_hook(our_state, bbox_state, rewards, next_states):
    if np.random.random() < our_state['alpha']:
        action = np.random.randint(0, infra.n_actions, 1)[0]
    else:
        action = np.argmax(rewards)

    our_state['writer'].append(bbox_state, action, rewards, next_states)
    return action




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", required=True, help="File name to save replay")
    parser.add_argument("--alpha", type=float, default=0.0, help="Fraction of random steps, by default=0.0")
    parser.add_argument("--depth", type=int, default=10, help="Depth to explore state")
    parser.add_argument("--gamma", type=float, default=0.95, help="Bellman's gamma")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    infra.setup_logging()
    infra.prepare_bbox()
    state = {
        'alpha': args.alpha,
        'gamma': args.gamma,
        'depth': args.depth,
        'writer': replays.ReplayWriter(args.output)
    }
    try:
        t = time()
        infra.bbox_checkpoints_loop(state, action_reward_hook, verbose=False)

        print("Replay generated in {duration}".format(
                duration=timedelta(seconds=time()-t)))
    finally:
        state['writer'].close()