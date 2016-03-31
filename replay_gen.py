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

    our_state['states'].append(bbox_state)
    our_state['actions'].append(action)
    our_state['rewards'].append(rewards)
    return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", required=True, help="File name to save replay")
    parser.add_argument("--alpha", type=float, default=0.0, help="Fraction of random steps, by default=0.0")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    infra.setup_logging()
    infra.prepare_bbox()
    state = {
        'alpha': args.alpha,
        'states': [],
        'actions': [],
        'rewards': [],
    }
    t = time()
    infra.bbox_checkpoints_loop(state, action_reward_hook, verbose=False)
    replays.save_replay(zip(state['states'], state['actions'], state['rewards']), args.output)
    print("Replay of length {len} generated in {duration}".format(
            len=len(state['states']), duration=timedelta(seconds=time()-t)))