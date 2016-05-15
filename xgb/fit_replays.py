import sys
sys.path.append("..")

import argparse
import numpy as np
import xgboost as xgb

from lib import features


def load_replays(file_names):
    l_states = []
    l_rewards = []

    for file_name in file_names:
        print "Load file %s" % file_name
        data = np.load(file_name)

        for row in data:
            state = features.reverse(*row[0])
#            state = features.to_dense(row[0])
            l_states.append(state)
            l_rewards.append(max(row[1]))

    states = np.array(l_states)
    rewards = np.array(l_rewards)

    return xgb.DMatrix(states, label=rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Test replay file to load")
    parser.add_argument("--rounds", type=int, default=100, help="Count of rounds to perform")
    parser.add_argument("files", nargs="+", help="Replay files to fit model")
    args = parser.parse_args()

    dtrain = load_replays(args.files)
    dtest = load_replays([args.test])

    params = {
        'silent': 1,
    }

    evallist = [(dtrain, "train"), (dtest, "test")]

    bst = xgb.train(params, dtrain, num_boost_round=args.rounds, evals=evallist)
    bst.save_model("test.xgb")