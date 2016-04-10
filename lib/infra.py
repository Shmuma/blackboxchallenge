import numpy as np
from time import time
from datetime import timedelta
import logging as log

from bbox import get_bbox

bbox = None
n_features = n_actions = None
test_level_loaded = None


def init(name=None):
    global bbox
    bbox = get_bbox(name)


def setup_logging(logfile=None, level=log.INFO):
    fmt = "%(asctime)s %(levelname)s %(message)s"
    if logfile is not None:
        log.basicConfig(filename=logfile, level=level, format=fmt)
    else:
        log.basicConfig(level=level, format=fmt)
    return log


def prepare_bbox(test_level=False):
    global n_features, n_actions, test_level_loaded

    if bbox is None:
        init()

    if bbox.is_level_loaded() and test_level_loaded == test_level:
        bbox.reset_level()
    else:
        if test_level:
            level_file = "levels/test_level.data"
        else:
            level_file = "levels/train_level.data"
        test_level_loaded = test_level
        bbox.load_level(level_file, verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()

    return n_features, n_actions


# Arguments:
# - our_state:      passed to all funcs
# - action_func:    calculate action for state
# - reward_func:    reward got from last action
def bbox_loop(our_state, action_func, reward_func, verbose=0, max_steps=None):
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

        if max_steps is not None and bbox.get_time() >= max_steps:
            break

    # last round, flush potential batch
    action_func(our_state, np.array(bbox.get_state()))
    reward_func(our_state, 0.0, last_round=True)

    log.info("Loop done in {duration}".format(duration=timedelta(seconds=time() - started)))
    return bbox.get_score()


def dig_all_actions(prev_score):
    """
    For all possible actions find all rewards and next states
    :param prev_score:
    :return: list with tuples (reward, stat)
    """
    chp_id = bbox.create_checkpoint()
    result = []

    for action in range(n_actions):
        bbox.do_action(action)
        r = bbox.get_score() - prev_score
        result.append((r, np.array(bbox.get_state())))
        bbox.load_from_checkpoint(chp_id)

    bbox.clear_all_checkpoints()
    return result


def bbox_checkpoints_loop(our_state, action_reward_func, verbose=0, max_steps=None):
    """
    Perform bbox loop using checkpoints function to explore all rewards
    :param our_state: state passed to all functions
    :param action_reward_func: function with signature (our_state, old_state, rewards, states) -> need to return action to take
    :param verbose: showe messages
    :return: total score earned
    """
    started = time()
    has_next = True

    while has_next:
        score = bbox.get_score()
        state = np.array(bbox.get_state())
        # explore state space
        rewards_states = dig_all_actions(score)
        rewards, states = zip(*rewards_states)
        # ask for best action
        action = action_reward_func(our_state, state, rewards, states)
        has_next = bbox.do_action(action)

        if verbose > 0:
            if bbox.get_time() % verbose == 0:
                log.info("{time}: Action {action} -> {reward} (rewards: {rewards}), "
                         "duration {duration}, score {score}".format(
                    time=bbox.get_time(), action=action, reward=bbox.get_score() - score,
                    rewards=rewards, duration=timedelta(seconds=time() - started),
                    score=bbox.get_score()
                ))

        if max_steps is not None and bbox.get_time() >= max_steps:
            break

    score = bbox.get_score()
    if verbose > 0:
        log.info("Loop done in {duration}, score {score}".format(
            duration=timedelta(seconds=time()-started), score=score
        ))

    return score