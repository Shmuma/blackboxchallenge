"""
External replays generator
"""
import os
import sys
sys.path.append("..")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from lib import infra, replays, net_light, features

import glob
import time
import argparse
import logging as log
import tensorflow as tf


REPLAYS_BATCH = 50000
REPLAYS_DIR = "replays"
MODELS_DIR = "models"


def last_model_file(run_name):
    models = []
    for f in glob.glob(os.path.join(MODELS_DIR, "model_" + run_name + "-*")):
        if f.endswith(".meta"):
            continue
        index = int(f.split("-")[1])
        models.append((index, f))
    if len(models) == 0:
        return None, None
    models.sort()
    return models[-1][1], models[-1][0]


def get_filename(index, start_time):
    return os.path.join(REPLAYS_DIR, "replay-{index:05}-t={start:04}k.npy".format(
        index=index, start=int(start_time/1000)
    ))


def make_summaries():
    vars = {
        'replay_score': tf.Variable(0.0, trainable=False, name="replay_score")
    }
    for name, var in vars.iteritems():
        tf.scalar_summary(name, var)
    return vars, tf.merge_all_summaries()


def write_summary(session, writer, score, vars, summary_t, model_step):
    if writer is None:
        return

    res, = session.run([summary_t], feed_dict={
        vars['replay_score']: score
    })

    writer.add_summary(res, model_step)
    writer.flush()


if __name__ == "__main__":
    if not os.path.exists(REPLAYS_DIR):
        os.mkdir(REPLAYS_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=400000, help="Maximim amount of steps to replay, default=400k")
    parser.add_argument("--batch", type=int, default=50000, help="Steps in every file, default=50k")
    parser.add_argument("--files", type=int, default=2, help="Count of files to maintain in replays dir, default=2")
    parser.add_argument("--name", required=True, help="Run name to track models")
    parser.add_argument("--oldname", help="Run name to use as fallback models source")
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha for generator, default=0.3")
    parser.add_argument("--cache", type=int, default=None, help="Cache game actions for given amout of steps, default=None")
    args = parser.parse_args()

    infra.setup_logging()
    infra.init()
    infra.prepare_bbox()

    last_model = None
    index_files = replays.find_replays(REPLAYS_DIR)
    index = 1 if len(index_files) == 0 else max(index_files)[0]+1
    log.info("Replay generator created, we have {files} files in replay dir, next index = {index}".format(
            files=len(index_files), index=index))

    state_t = tf.placeholder(tf.float32, (1, features.RESULT_N_FEATURES))
    qvals_t = net_light.make_forward_net(state_t, features.RESULT_N_FEATURES)

    with tf.Session() as session:
        # create replay generator
        replay_generator = replays.ReplayGenerator(args.batch, session, state_t, qvals_t,
                                                   alpha=args.alpha, reset_after_steps=args.max)
        saver = tf.train.Saver()

        # summary_writers = {}
        # for step in set(range(args.batch, args.max, args.batch) + [args.max]):
        #     summary_writers[step] = tf.train.SummaryWriter("logs/" + args.name + "-scores-%04dK" % (int(step/1000)))
        #
        # summ_vars, summary_t = make_summaries()

        while True:
            # discover and load latest model file
            model, model_step = last_model_file(args.name)
            if model is None and args.oldname is not None:
                model, model_step = last_model_file(args.oldname)
            if model is None:
                log.info("No model file exists, sleep")
                time.sleep(60)
                continue
            if model != last_model:
                log.info("New model file discovered, loading {model}".format(model=model))
                last_model = model
                saver.restore(session, last_model)

            index_files = replays.find_replays(REPLAYS_DIR)
            if len(index_files) >= args.files:
                time.sleep(60)
                continue

            start_time = infra.bbox.get_time()
            log.info("Will create next batch from bbox_time={bbox_time}, score={score:.3f}".format(
                    bbox_time=start_time, score=infra.bbox.get_score()))

            batch = replay_generator.next_batch()
            file_name = get_filename(index, start_time)
            replays.save_replay_batch(file_name, batch)
            log.info("Generated, score={score}, data saved in {file}".format(
                    score=infra.bbox.get_score(), file=file_name))
            score = max(replay_generator.score, replay_generator.last_score)
            # write_summary(session, summary_writers.get(start_time + args.batch), score, summ_vars, summary_t, model_step)
            index += 1
