import sys
sys.path.append("..")

import argparse
import numpy as np
import xgboost as xgb

from lib import features


def load_replay(file_name):
    l_states = []
    l_rewards = []

    print "Load file %s" % file_name
    data = np.load(file_name)

    for row in data:
        orig_state = features.reverse(*row[0])
        l_states.append(orig_state)
        l_rewards.append(max(row[1]))

    states = np.array(l_states)
    rewards = np.array(l_rewards)

    return xgb.DMatrix(states, label=rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Test replay file to load")
    parser.add_argument("files", nargs="+", help="Replay files to fit model")
    args = parser.parse_args()

    print args.files

    dtrain = load_replay(args.files[0])
    dtest = load_replay(args.test)

    params = {

    }

    evallist = [(dtrain, "train"), (dtest, "test")]

    bst = xgb.train(params, dtrain, num_boost_round=100, evals=evallist)
    bst.save_model("test.xgb")