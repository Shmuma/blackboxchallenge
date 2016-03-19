import interface as bbox
import numpy as np
from skimage import io
import matplotlib.pylab as pl

n_features = n_actions = -1


def prepare_bbox():
    global n_features, n_actions

    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()


def get_action_by_state(state):
#    return np.random.randint(0, 4)
    return 0

if __name__ == "__main__":
    has_next = 1
    prepare_bbox()
    prev_score = bbox.get_score()
    steps = 0

    states = []

    while has_next and steps < 100:
        state = bbox.get_state()
        states.append(state)
        v = map(lambda f: "%.2f" % abs(f), state)
        print " ".join(v)
        action = get_action_by_state(state)
        has_next = bbox.do_action(action)
        score = bbox.get_score()
        prev_score = score
        steps += 1

#    bbox.finish(verbose=1)
    print "Total score: %f" % prev_score
    print "Total steps: %d" % steps

    img = np.stack(states)
    img -= img.mean()
    img /= img.max() - img.min()
    print img.std()
    io.imsave("image.png", img)