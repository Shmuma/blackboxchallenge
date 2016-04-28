"""
Tool does automatical detection of stripes in input features
"""
import sys
sys.path.append("..")
import argparse

import numpy as np
import logging as log

from lib import infra

EPSILON = 1e-5

# feature 25: Looks strange
# definetely striped features: 0, 5-12, 23, 24, 35
#EXCLUDE_FEATURES = {1, 2, 3, 4, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, }
INCLUDE_FEATURES = {0, 5, 6, 7, 8, 9, 10, 11, 12, 23, 24, 35}

def lookup_stripe(stripes, value, eps=EPSILON):
    """
    Lookup stripe in stripes array. If not found, new stripe is added
    :param stripes:
    :param value:
    :param eps:
    :return: index
    """
    for idx, s_val in enumerate(stripes):
        if abs(s_val-value) < eps:
            return idx
    stripes.append(value)
    return len(stripes)-1


def stripify(data, eps=EPSILON):
    """
    Build a list of stripes from data array
    :param data:
    :return: dict with stripe position and count of data points in it
    """
    stripes = []
    res = {}

    for idx, val in enumerate(data):
        s_idx = lookup_stripe(stripes, val, eps)
        s_val = stripes[s_idx]
        res[s_val] = res.get(s_val, 0) + 1

    log.info("Raw stripes len=%d", len(res))

    return res


def try_stripes(start, delta, stripes_counts, up_levels, eps=EPSILON, count=60):
    """
    Try to build 60-levels stripes from starting point.
    :param start:
    :param delta:
    :param stripes_counts:
    :param up_levels: levels above and including starting point
    :return: tuple with matched and unmatched samples count
    """
    matched, missed = 0, 0
    last_level = start + delta*count

    axis = [(level, False) for level in up_levels if level < last_level]
    axis += [(start + i*delta, True) for i in range(count)]
    axis.sort()

    res = []
    for pt in axis:
        if len(res) == 0:
            res.append(pt)
            continue

        l = res.pop()
        if abs(l[0] - pt[0]) < eps:
            if l[1]:
                res.append((pt[0], True))
            else:
                res.append((l[0], pt[1]))
        else:
            res.append(l)
            res.append(pt)

    for pt in res:
        samples_count = stripes_counts.get(pt[0], 0)
        if pt[1]:
            matched += samples_count
        else:
            missed += samples_count

    return matched, missed


def guess_delta(levels, eps=EPSILON):
    d = np.diff(levels)
    d.sort()
    vals = []

    for v in d:
        if len(vals) == 0 or abs(np.mean(vals) - v) < eps:
            vals.append(v)
        else:
            break

    return np.mean(vals)


def group_stripes(stripes_dict, eps=EPSILON, count=60):
    """
    Try to find stripes placement for existing levels. I know that usual stripes count is 60 or 1, so I check
    all possible starting values + deltas and greedily assigning levels using next item's delta
    :param stripes_dict:
    :return:
    """
    levels = list(stripes_dict.keys())
    levels.sort()

    skip = None
    for idx, start_val in enumerate(levels):
        if len(levels) - idx < 3:
            break
        if skip is not None and start_val < skip:
            continue

        part_levels = levels[idx:idx+count]
        delta = guess_delta(part_levels, eps=eps)

        # simple check for stripe with one value
        next_delta = part_levels[1] - part_levels[0]
        if next_delta / delta > 50:
            log.info("idx={idx}, level={level:.10f}: explained={explained}".format(
                idx=idx, level=start_val, explained=stripes_dict[start_val]
            ))
        else:
            explained, missed = try_stripes(start_val, delta, stripes_dict, part_levels, eps=eps, count=count)
            if missed == 0:
                skip = start_val + delta*count
                log.info("idx={idx}, start={level:.10f}, stop={stop:.10f}, delta={delta:.10f}: explained={explained}, missed={missed}".format(
                         idx=idx, level=start_val, stop=start_val + delta*(count-1),
                         delta=delta, explained=explained, missed=missed
                ))
            else:
                skip = None
                log.info("* idx={idx}, level={level:.10f}, delta={delta:.10f}: explained={explained}, missed={missed}".format(
                    idx=idx, level=start_val, delta=delta, explained=explained, missed=missed
                ))


def bbox_action_hook(st, state, rewards, next_states):
    action = np.random.randint(0, infra.n_actions, 1)[0]

    st['states'] = np.concatenate([st['states'], [state], next_states])
#    st['stripes'] = np.concatenate([st['stripes'], [[None] * infra.n_features] * 5])

    st['step'] += 1
    if st['step'] % 1000 == 0:
        log.info("{step}: states={states}".format(
                step=st['step'], states=st['states'].shape[0]))

    if st['step'] % 100000 == 0:
        for feat in range(infra.n_features):
            if feat not in INCLUDE_FEATURES:
                log.info("Feature %d excluded", feat)
                continue
            log.info("Process feature %d", feat)
            stripes = stripify(st['states'][:, feat])
            group_stripes(stripes)

        sys.exit()

    return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Use test levels")
    args = parser.parse_args()

    infra.setup_logging()
    infra.prepare_bbox()

    state = {
        'step': 0,
        'states': np.zeros((0, infra.n_features)),
        'stripes': np.zeros((0, infra.n_features)),
    }

    infra.bbox_checkpoints_loop(state, bbox_action_hook)

