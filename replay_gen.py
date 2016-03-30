"""
Replay generator
"""
import numpy as np
import argparse
import cPickle as pickle

from lib import infra


def action_hook(our_state, bbox_state):
    action = np.random.randint(0, infra.n_actions, 1)[0]
    our_state['states'].append(bbox_state)
    our_state['actions'].append(action)
    return action


def reward_hook(our_state, reward, last_round):
    our_state['rewards'].append(reward)


def save_replay(our_state, output_file):
    with open(output_file, "wb") as fd:
        pickle.dump(zip(our_state['states'], our_state['actions'], our_state['rewards']), fd)


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
    infra.bbox_loop(state, action_hook, reward_hook, verbose=False)
    save_replay(state, args.output)