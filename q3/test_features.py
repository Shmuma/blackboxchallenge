"""
Experiment with features transformation on converged model
"""
import sys, os
sys.path.append("..")
# comment this to enable GPU TF
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import csv
import argparse
import tensorflow as tf
import logging as log
import numpy as np
from time import time
from datetime import timedelta

from lib import infra, net, run_bbox, features

ALPHA = 0.05


def run_step(ctx, name, transformer=None):
    log.info("Starting step: {step}".format(step=name))
    t = time()

    scores = []
    for round in xrange(ctx['rounds']):
        score, _ = run_bbox.test_performance(ctx['session'], ctx['state_t'], ctx['qvals_t'],
                                             alpha=ALPHA, max_steps=ctx['steps'], test_level=True,
                                             feats_tr_post=transformer)
        scores.append(score)

    res = {
        'mean': np.mean(scores),
        'max': np.max(scores),
        'min': np.min(scores),
        'std': np.std(scores)
    }
    log.info("Done in {duration}, result={res}".format(res=res, duration=timedelta(seconds=time()-t)))
    return res


def run_test_blank_one(context, res_baseline):
    """
    Perform test by blanking each of 36 features and check result
    :param context:
    :param res_baseline:
    :return:
    """
    def get_feat_blanker(feat_idx):
        """
        Return function which blanks this particular feature completely
        :param feat_idx:
        :return: function
        """
        from_idx = 0
        for f in range(feat_idx):
            from_idx += features.sizes.get(f, 1)
        to_idx = from_idx + features.sizes.get(feat_idx, 1)

        def _blanker(features):
#            log.info("Blanker from={from_idx}, to={to_idx}".format(from_idx=from_idx, to_idx=to_idx))
#            log.info("Before: " + str(features.nonzero()))
            features[from_idx:to_idx] = 0.0
#            log.info("After: " + str(features.nonzero()))
            return features

        return _blanker

    res = {}

    for blank_feat in xrange(features.ORIGIN_N_FEATURES):
        res[blank_feat] = run_step(context, "Blank feature {feature}".format(feature=blank_feat),
                 transformer=get_feat_blanker(blank_feat))

    if context['csv'] is not None:
        with open(context['csv'], "w+") as fd:
            writer = csv.writer(fd)
            writer.writerow(["Blanked feature", "Mean", "Max", "Min", "StdDev"])
            writer.writerow([-1, res_baseline['mean'], res_baseline['max'], res_baseline['min'], res_baseline['std']])

            for feat in sorted(res.keys()):
                writer.writerow([feat, res[feat]['mean'], res[feat]['max'], res[feat]['min'], res[feat]['std']])

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model file name to load")
    parser.add_argument("--test", required=True, choices=['blank_one'], help="Type of test to perform")
    parser.add_argument("--steps", type=int, default=20000, help="Count of steps")
    parser.add_argument("--rounds", type=int, default=10, help="Count of rounds for every test")
    parser.add_argument("--csv", type=str, help="Report file to be created, default=no file")
    args = parser.parse_args()

    infra.setup_logging()
    infra.prepare_bbox()
    n_features = features.transformed_size()

    state_t = tf.placeholder(tf.float32, (None, n_features))
    qvals_t = net.make_forward_net_v3(state_t, True, n_features, dropout_keep_prob=1.0)
    saver = tf.train.Saver(var_list=dict(net.get_v2_vars(trainable=True)).values())

    with tf.Session() as session:
        session.run([tf.initialize_all_variables()])
        log.info("Loading model from %s", args.model)
        saver.restore(session, args.model)

        context = {
            'session': session,
            'state_t': state_t,
            'qvals_t': qvals_t,
            'rounds': args.rounds,
            'steps': args.steps,
            'csv': args.csv,
        }

        log.info("Initialisation done, running test '{test}', rounds={rounds}, steps={steps}".format(
                test=args.test, rounds=args.rounds, steps=args.steps))
        res_baseline = run_step(context, name="baseline (no features tweaking)")

        if args.test == "blank_one":
            run_test_blank_one(context, res_baseline)