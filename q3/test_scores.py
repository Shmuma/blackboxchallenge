"""
Tool which monitors model save directory for new files and test this model against train and test data
"""
import sys
import os
sys.path.append("..")
# comment this to enable GPU TF
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import subprocess
import argparse
import json
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
        if name.startswith("train"):
            tf.scalar_summary(name, v, collections=["train"])
        else:
            tf.scalar_summary(name, v, collections=["test"])

    res['test'] = tf.merge_all_summaries(key="test")
    res['train'] = tf.merge_all_summaries(key="train")
    return res


def write_summaries(writer, step, session, summ, test_mode, scores):
    if test_mode:
        key = "test"
        feed = {
            summ['test_mean']: np.mean(scores),
            summ['test_min']: float(min(scores)),
            summ['test_max']: float(max(scores)),
            summ['test_std']: np.std(scores),
        }
    else:
        key = "train"
        feed = {
            summ['train_mean']: np.mean(scores),
            summ['train_min']: float(min(scores)),
            summ['train_max']: float(max(scores)),
            summ['train_std']: np.std(scores),
        }

    summ_res, = session.run([summ[key]], feed_dict=feed)
    writer.add_summary(summ_res, step)
    writer.flush()


def process_slave(args):
    """
    Simulate one run
    :param args:
    :return:
    """
    infra.prepare_bbox(args.test)
    n_features = features.transformed_size()

    state_t = tf.placeholder(tf.float32, (None, n_features))
    qvals_t = net.make_forward_net(state_t, True, n_features, dropout_keep_prob=1.0)

    saver = tf.train.Saver(var_list=dict(net.get_vars(trainable=True)).values())

    with tf.Session() as session:
        session.run([tf.initialize_all_variables()])

        model_file = args.slave
        saver.restore(session, model_file)

        scores = []
        t = time()
        for round in xrange(args.rounds):
            score, _ = run_bbox.test_performance(session, state_t, qvals_t,
                                                 alpha=args.alpha, max_steps=args.steps,
                                                 test_level=False)
            scores.append(score)

        res = {
            'model': args.slave,
            'test': args.test,
            'scores': scores,
            'step': args.start,
            'time': time() - t
        }
        print "Result={res}".format(res=json.dumps(res))


def make_slave_args(args, step, model_file, test):
    res = [
        "python",
        sys.argv[0],
        "--slave", model_file,
        "--name", args.name,
        "--rounds", str(args.rounds),
        "--alpha", str(args.alpha),
        "--steps", str(args.steps),
        "--start", str(step),
    ]

    if test:
        res.append("--test")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of the run to watch")
    parser.add_argument("--steps", type=int, default=20000, help="Limit amount of steps")
    parser.add_argument("--rounds", type=int, default=10, help="Amount of rounds to perform")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha value for testing")
    parser.add_argument("--start", type=int, default=0, help="Global step to start processing")
    parser.add_argument("--once", action="store_true", default=False, help="Loop over model files once and exit")
    parser.add_argument("--suffix", default="score", help="Summary suffix to run name")
    parser.add_argument("--parallel", type=int, default=4, help="Max count of child processes to start")
    parser.add_argument("--slave", default=None, help="Used to start slave processes, do not use")
    parser.add_argument("--test", default=False, action="store_true", help="Used in slave mode")
    args = parser.parse_args()

    if args.slave is not None:
        process_slave(args)
    else:
        log = infra.setup_logging()
        start = args.start

        with tf.Session() as session:
            summs = make_summaries()
            session.run([tf.initialize_all_variables()])

            log.info("Initialisation done, looking for new model files")
            summary_writer = tf.train.SummaryWriter("logs/" + args.name + "-" + args.suffix)
            models_to_process = []
            slaves = []
            done_slaves = []

            while True:
                for step in discover_new_steps(start, args.name):
                    log.info("Found new model for step %d" % step)
                    model_file = get_model_path(args.name, str(step))
                    models_to_process += [(model_file, step, False), (model_file, step, True)]
                    start = step

                # check for terminated slave processes
                running_slaves = []
                for p in slaves:
                    r = p.poll()
                    if r is not None:
                        stdout, _ = p.communicate(None)
                        for l in stdout.split("\n"):
                            if l.startswith("Result="):
                                res = json.loads(l.split("=")[1])
                                log.info("Slave for model %s, test=%s done in %s",
                                         res['model'], res['test'], timedelta(seconds=res['time']))
                                done_slaves.append(res)
                    else:
                        running_slaves.append(p)
                slaves = running_slaves

                if len(slaves) == 0:
                    # save results sorted by step
                    done_slaves.sort(key=lambda r: r['step'])
                    for res in done_slaves:
                        write_summaries(summary_writer, res['step'], session, summs, res['test'], res['scores'])
                    done_slaves = []

                    # enqueue more slaves
                    while len(slaves) < args.parallel and len(models_to_process) > 0:
                        model_file, step, test_mode = models_to_process.pop(0)
                        log.info("Started subprocesses to handle model file %s, test=%s", model_file, test_mode)
                        p = subprocess.Popen(make_slave_args(args, step, model_file, test=test_mode),
                                             stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                        slaves.append(p)

                sleep(10)
