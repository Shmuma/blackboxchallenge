import numpy as np

HISTORY = 10
TIME_COUNT = 10
ORIGIN_N_FEATURES = 36
RESULT_N_FEATURES = 3 + HISTORY * 2 + TIME_COUNT


def transform(state, reward_history, actions_history, bbox_time):
    """
    All features except last transfomed into bits value, last left as is
    :param state:
    :return: transformed numpy vector
    """
    vals = state[[1, 2, -1]]
    time = np.zeros((TIME_COUNT, ))
    time[bbox_time % TIME_COUNT] = 1.0

    if len(reward_history) < HISTORY:
        reward_history = [0.0] * (HISTORY - len(reward_history)) + reward_history

    if len(actions_history) < HISTORY:
        actions_history = [0.0] * (HISTORY - len(actions_history)) + actions_history

    return np.concatenate([vals, time, reward_history[-HISTORY:], actions_history[-HISTORY:]])


def push_history(buffer, item):
    buffer.append(item)
    return buffer[-HISTORY:]
