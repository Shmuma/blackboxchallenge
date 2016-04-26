"""
Tool which monitors model save directory for new files and test this model against train and test data
"""
import sys
sys.path.append("..")
import argparse

import os
import glob
from time import time, sleep
from datetime import timedelta
import tensorflow as tf
import numpy as np

from lib import infra, net, run_bbox, features

MODEL_DIR = "models"


def get_model_path(run_name, step):
    return os.path.join(MODEL_DIR, "model_" + run_name + "-" + step)


def discover_new_steps(start, run_name):
    steps = []
    for name in glob.glob(get_model_path(run_name, "*")):
        if name.endswith(".meta"):
            continue
        v = os.path.basename(name).split("-")[-1]
        v = int(v)
        if v > start:
            steps.append(int(v))
    steps.sort()
    return steps


def make_summaries():
    res = {
        'train_mean': tf.Variable(0.0, name='train_mean'),
        'train_max': tf.Variable(0.0, name='train_max'),
        'train_min': tf.Variable(0.0, name='train_min'),
        'train_std': tf.Variable(0.0, name='train_std'),
        'test_mean': tf.Variable(0.0, name='test_mean'),
        'test_max': tf.Variable(0.0, name='test_max'),
        'test_min': tf.Variable(0.0, name='test_min'),
        'test_std': tf.Variable(0.0, name='test_std'),
    }

    for name, v in res.iteritems():
        tf.scalar_summary(name, v, collections=["train_test"])

    res['all'] = tf.merge_all_summaries(key="train_test")
    return res


def write_summaries(writer, step, session, summ, train_scores, test_scores):
    feed = {
        summ['train_mean']: np.mean(train_scores),
        summ['train_min']: float(min(train_scores)),
        summ['train_max']: float(max(train_scores)),
        summ['train_std']: np.std(train_scores),

        summ['test_mean']: np.mean(test_scores),
        summ['test_min']: float(min(test_scores)),
        summ['test_max']: float(max(test_scores)),
        summ['test_std']: np.std(test_scores),
    }

    summ_res, = session.run([summ['all']], feed_dict=feed)
    writer.add_summary(summ_res, step)
    writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of the run to watch")
    parser.add_argument("--steps", type=int, default=20000, help="Limit amount of steps")
    parser.add_argument("--rounds", type=int, default=10, help="Amount of rounds to perform")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha value for testing")
    parser.add_argument("--start", type=int, default=0, help="Global step to start processing")
    parser.add_argument("--once", action="store_true", default=False, help="Loop over model files once and exit")
    args = parser.parse_args()

    log = infra.setup_logging()
    infra.prepare_bbox()
    n_features = features.transformed_size()

    state_t = tf.placeholder(tf.float32, (None, n_features))
    qvals_t = net.make_forward_net(state_t, True, n_features, dropout_keep_prob=1.0)

    saver = tf.train.Saver(var_list=dict(net.get_vars(trainable=True)).values())
    start = args.start

    with tf.Session() as session:
        session.run([tf.initialize_all_variables()])

        log.info("Initialisation done, looking for new model files")
        summary_writer = tf.train.SummaryWriter("logs/" + args.name + "-score")
        summs = make_summaries()

        while True:
            for step in discover_new_steps(start, args.name):
                t = time()
                log.info("Found new model for step %d" % step)
                model_file = get_model_path(args.name, str(step))

                log.info("Loading model from %s", model_file)
                saver.restore(session, model_file)

                log.info("Run %d rounds on train data", args.rounds)

                train_scores = []
                for round in xrange(args.rounds):
                    score, _ = run_bbox.test_performance(session, state_t, qvals_t,
                                                         alpha=args.alpha, max_steps=args.steps,
                                                         test_level=False)
#                    log.info("Round {round}: score={score}".format(round=round+1, score=score))
                    train_scores.append(score)
                log.info("Test done, averaged score={score}, min={min_score}, max={max_score}, std={std}".format(
                    score=np.mean(train_scores), min_score=min(train_scores), max_score=max(train_scores), std=np.std(train_scores)
                ))

                log.info("Run %d rounds on test data", args.rounds)
                test_scores = []
                for round in xrange(args.rounds):
                    score, _ = run_bbox.test_performance(session, state_t, qvals_t,
                                                         alpha=args.alpha, max_steps=args.steps,
                                                         test_level=True)
#                    log.info("Round {round}: score={score}".format(round=round+1, score=score))
                    test_scores.append(score)
                log.info("Test done, averaged score={score}, min={min_score}, max={max_score}, std={std}".format(
                    score=np.mean(test_scores), min_score=min(test_scores), max_score=max(test_scores), std=np.std(test_scores)
                ))

                log.info("Test done in %s, write to log", timedelta(seconds=time() - t))
                write_summaries(summary_writer, step, session, summs, train_scores, test_scores)
                start = step
            if args.once:
                break
            sleep(60)