import logging as log
import numpy as np

from _gridworld import GridWorldState

_state = GridWorldState(size=(2, 2), start=(0, 0), end=(1, 1), rewards=[(1, 1, 10)])


def load_level(filename, verbose=True):
    log.info("Request to load level from {file}, ignored".format(file=filename))


def finish(verbose=True):
    log.info("Finish level")
    score = _state.score
    _state.reset()
    return score

def is_level_loaded():
    return True

def get_state():
    return _state.state()

def get_time():
    return _state.time

def get_score():
    return _state.score

def do_action(action):
    old = str(_state)
    next = _state.do_action(action)
    log.info("Action {action}: {old}=>{new}".format(
            action=action, old=old, new=str(_state)))
    return next

def get_num_of_features():
    return _state.num_features()

def get_num_of_actions():
    return _state.num_actions()