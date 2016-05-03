import interface as bbox

import time
import datetime
import numpy as np
from net_light import calc_qvals, load_weights
from features import transform, to_dense

VERBOSE = True
REPORT_INTERVAL = 10000
start_t = time.time()
last_t = None
MODEL_FILE = "model_t38r1-400000.npy"

network_weights = {}
n_features = n_actions = max_time = -1


def get_action_by_state(state):
    global last_t, start_t
    if VERBOSE:
        if bbox.get_time() % REPORT_INTERVAL == 0:
            msg = "total=%s" % (datetime.timedelta(seconds=time.time() - start_t))
            if last_t is not None:
                d = time.time() - last_t
                speed = REPORT_INTERVAL / d
                msg += ", time=%s, speed=%.3f steps/s" % (datetime.timedelta(seconds=d), speed)
            print "Step=%d, score=%.2f, %s" % (bbox.get_time(), bbox.get_score(), msg)
            last_t = time.time()
    dense_state = to_dense(transform(state))
    qvals = calc_qvals(network_weights, dense_state)
    return np.argmax(qvals)


# Participants do not have to modify code below, but it could be useful to understand what this code does.
def prepare_bbox():
    global n_features, n_actions, max_time

    # Reset environment to the initial state, just in case
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        # Load the game level
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()


def run_bbox(verbose=False):
    has_next = 1

    # Prepare environment - load the game level
    prepare_bbox()

    while has_next:
        # Get current environment state
        state = bbox.get_state()

        # Choose an action to perform at current step
        action = get_action_by_state(state)

        # Perform chosen action
        # Function do_action(action) returns False if level is finished, otherwise returns True.
        has_next = bbox.do_action(action)

    # Finish the game simulation, print earned reward
    # While submitting solutions, make sure that you do call finish()
    bbox.finish(verbose=1)


if __name__ == "__main__":
    network_weights = load_weights(MODEL_FILE)
    run_bbox(verbose=False)