import interface as bbox
import numpy as np
from time import time
from datetime import timedelta
import logging as log

n_features = n_actions = None


def setup_logging(logfile=None, level=log.INFO):
    fmt = "%(asctime)s %(levelname)s %(message)s"
    if logfile is not None:
        log.basicConfig(filename=logfile, level=level, format=fmt)
    else:
        log.basicConfig(level=level, format=fmt)
    return log


def prepare_bbox():
    global n_features, n_actions

    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()

    return n_features, n_actions


# Arguments:
# - our_state:      passed to all funcs
# - action_func:    calculate action for state
# - reward_func:    reward got from last action
def bbox_loop(our_state, action_func, reward_func, verbose=0, max_time=0):
    prev_score = bbox.get_score()

    started = time()
    has_next = True

    while has_next:
        state = np.array(bbox.get_state())
        action = action_func(our_state, state)
        has_next = bbox.do_action(action)

        score = bbox.get_score()
        reward = score - prev_score
        prev_score = score

        reward_func(our_state, reward, last_round=False)

        if verbose:
            if bbox.get_time() % verbose == 0:
                log.info("%d: Action %d -> %f, duration %s, score %f",
                    bbox.get_time(), action, reward,
                    timedelta(seconds=time() - started), score)

    # last round, flush potential batch
    action_func(our_state, np.array(bbox.get_state()))
    reward_func(our_state, 0.0, last_round=True)

    log.info("Loop done in {duration}".format(duration=timedelta(seconds=time() - started)))
