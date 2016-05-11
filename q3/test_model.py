"""
The same logic as submit/bot.py, but with more options
"""
import os
import sys
sys.path.append("..")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from lib import interface as bbox

import time
import argparse
import datetime
import numpy as np
import tensorflow as tf
from lib import features, net_light

VERBOSE = True
TEST_LEVEL = True
REPORT_INTERVAL = 10000
start_t = time.time()
last_t = None

n_features = n_actions = max_time = -1

state_t = tf.placeholder(tf.float32, (1, features.RESULT_N_FEATURES))
qvals_t = net_light.make_forward_net(state_t, features.RESULT_N_FEATURES)

global_session = None
args = None
cached_step = None
cached_counter = 0


def get_action_by_state(state):
    global last_t, start_t
    global cached_step, cached_counter

    if VERBOSE:
        if bbox.get_time() % REPORT_INTERVAL == 0:
            msg = "total=%s" % (datetime.timedelta(seconds=time.time() - start_t))
            if last_t is not None:
                d = time.time() - last_t
                speed = REPORT_INTERVAL / d
                msg += ", time=%s, speed=%.3f steps/s" % (
                    datetime.timedelta(seconds=d), speed
                )
            print "Step=%d, score=%.2f, %s" % (bbox.get_time(), bbox.get_score(), msg)
            sys.stdout.flush()
            last_t = time.time()

    cached_counter -= 1
    if cached_counter > 0:
        return cached_step

    sparse_state = features.transform(state)
    dense_state = features.to_dense(sparse_state)

    qvals, = global_session.run([qvals_t], feed_dict={
        state_t: [dense_state]
    })
    cached_step = np.argmax(qvals)
    cached_counter = args.cache
    return cached_step


# Participants do not have to modify code below, but it could be useful to understand what this code does.
def prepare_bbox():
    global n_features, n_actions, max_time

    # Reset environment to the initial state, just in case
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        # Load the game level
        if args.test:
            level_file = "../levels/test_level.data"
        else:
            level_file = "../levels/train_level.data"
        bbox.load_level(level_file, verbose=1)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=int, default=1, help="How many steps to cache")
    parser.add_argument("--test", action="store_true", default=False, help="Use test levels")
    parser.add_argument("model", help="Model file to load")
    args = parser.parse_args()

    print "Arguments: %s" % args

    with tf.Session() as session:
        global_session = session
        saver = tf.train.Saver()
        saver.restore(session, args.model)

        run_bbox(verbose=False)
