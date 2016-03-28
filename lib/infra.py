import interface as bbox
import numpy as np

n_features = n_actions = None


def prepare_bbox():
    global n_features, n_actions

    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()

    return n_features, n_actions


def bbox_loop(our_state, get_action_func, learn_func):
    prev_score = bbox.get_score()

    has_next = True
    prev_state = np.array(bbox.get_state())

    while has_next:
        action = get_action_func(our_state)
        has_next = bbox.do_action(action)
        score = bbox.get_score()
        reward = score - prev_score
        prev_score = score

        state = np.array(bbox.get_state())

        print "%d: Action %d -> %f" % (bbox.get_time(), action, reward)

        # do q-learning stuff
        if prev_state is not None:
            learn_func(our_state, prev_state, action, reward, state)

        # update our states
        prev_state = state
