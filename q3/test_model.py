import sys
sys.path.append("..")
import argparse

from lib import infra, net, run_bbox, features

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="File with model to restore")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha value for testing, default=0.05")
    parser.add_argument("--steps", type=int, default=None, help="Limit amount of steps, default=no limit")
    parser.add_argument("--rounds", type=int, default=1, help="Count of rounds to perform, default=1")
    parser.add_argument("--quiet", action="store_true", default=False, help="Do not show intermediate rounds")
    parser.add_argument("--verbose", type=int, default=None, help="Show progress of testing every N steps")
    parser.add_argument("--test", action="store_true", default=False, help="Use test levelfile")
    args = parser.parse_args()

    log = infra.setup_logging()
    infra.prepare_bbox()

    n_features = features.transformed_size()
    state_t = tf.placeholder(tf.float32, (None, n_features), name="state")
    qvals_t = net.make_forward_net(state_t, True, n_features, dropout_keep_prob=1.0)

    saver = tf.train.Saver(var_list=dict(net.get_vars(trainable=True)).values())

    with tf.Session() as session:
        session.run([tf.initialize_all_variables()])
        saver.restore(session, args.model)

        log.info("Start testing of model {model} with alpha={alpha} for {steps} steps for {rounds} rounds".format(
            model=args.model, alpha=args.alpha, rounds=args.rounds,
            steps="all" if args.steps is None else args.steps
        ))
        scores = []

        for round in xrange(args.rounds):
            score, _ = run_bbox.test_performance(session, state_t, qvals_t,
                                                 alpha=args.alpha, max_steps=args.steps,
                                                 verbose=args.verbose, test_level=args.test)
            if not args.quiet:
                log.info("Round {round}: score={score}".format(round=round+1, score=score))

            scores.append(score)

        if args.rounds > 1:
            log.info("Test done, averaged score={score}, min={min_score}, max={max_score}, std={std}".format(
                score=np.mean(scores), min_score=min(scores), max_score=max(scores), std=np.std(scores)
            ))
