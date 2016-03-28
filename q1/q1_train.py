import sys
sys.path.append("..")

from lib import infra
import numpy as np
import tensorflow as tf


def random_action(our_state):
    return np.random.randint(0, infra.n_actions, 1)


def learn_q(our_state, prev_state, action, reward, new_state):
    # calculate Q* from:
    # - reward we've got
    # - best Q-value from new state
    pass


if __name__ == "__main__":
    np.random.seed(42)
    infra.prepare_bbox()
    infra.bbox_loop(None, random_action, learn_q)
    infra.bbox.finish(verbose=1)
