"""
Test module with very simple (logging world)
"""
import logging as log
import numpy as np

def load_level(filename, verbose=True):
    log.info("Loading level from {file}".format(file=filename))


def finish(verbose=True):
    log.info("Finish level")
    return 0.0

def is_level_loaded():
    return True

def get_state():
    return np.zeros((36,))

def get_time():
    return 0

def get_score():
    return 0.0

def do_action(action):
    log.info("Action {action}".format(action=action))
    return False

def get_num_of_features():
    return 36

def get_num_of_actions():
    return 4

