import sys
import numpy as np
import pandas as pd
import matplotlib.pylab as pl
import seaborn as sb


def feature_data(index, seed):
    return "../tests/out/seed={seed}/{index:02d}-reward-state.txt".format(seed=seed, index=index)


def load_feature(index, seed=42):
    data_file = feature_data(index, seed)
    print "Loading from " + data_file
    return pd.read_csv(data_file, header=None)


def overview(feat):
    print feat.describe()
    sys.stdout.flush()
    pl.scatter(feat[0], feat[1], marker='.')
    pl.show()


def bounds_feat(feat, bounds, no_zero_rw=False):
    res = (feat[1] >= bounds[0]) & (feat[1] <= bounds[1])
    if no_zero_rw:
        res = res & (abs(feat[0]) > 1e-5)
    return res


def deltas(arr):
    prev = None
    res = []
    for v in arr:
        if prev is not None:
            res.append(v - prev)
        prev = v
    return res

def scatter_mask(feat, mask):
    pl.scatter(feat[0][mask], feat[1][mask], marker='.')
    pl.show()