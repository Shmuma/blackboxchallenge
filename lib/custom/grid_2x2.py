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
#    log.info("Action {action}({aname}): {old} => {new}".format(
#            action=action, aname=_state.action_name(action),
#            old=old, new=str(_state)))
    return next

def get_num_of_features():
    return _state.num_features()

def get_num_of_actions():
    return _state.num_actions()

def reset_level():
    _state.reset()
#    log.info("Reset level to: {state}".format(state=str(_state)))

def create_checkpoint():
    return _state.create_checkpoint()

def load_from_checkpoint(chp_id):
    _state.load_from_checkpoint(chp_id)

def clear_all_checkpoints():
    _state.clear_all_checkpoints()


def _all_states():
    return _state._all_states()

def _describe_state(state):
    return _state._describe_state(state)