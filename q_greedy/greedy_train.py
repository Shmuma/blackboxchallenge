import sys
sys.path.append("..")

from lib import infra, net
from time import time
from datetime import timedelta
import numpy as np
import tensorflow as tf

N_STATE = 36
N_ACTIONS = 4

BATCH_SIZE = 100
REPORT_ITERS = 100
SAVE_MODEL_ITERS = 1000

def make_greedy_readers(file_prefix):
    """
    Return states and qvals tensors
    :param file_prefix:
    :return:
    """
    states_reader = tf.FixedLengthRecordReader(N_STATE * 4)
    qvals_reader = tf.FixedLengthRecordReader(N_ACTIONS * 4)
    _, states = states_reader.read(tf.train.string_input_producer([file_prefix + ".states"]))
    _, qvals = qvals_reader.read(tf.train.string_input_producer([file_prefix + ".rewards"]))

    states = tf.decode_raw(states, tf.float32, name="decode_states")
    states = tf.reshape(states, (N_STATE, ), name="reshape_states")
    qvals = tf.decode_raw(qvals, tf.float32, name="decode_qvals")
    qvals = tf.reshape(qvals, (N_ACTIONS, ), name="reshape_qvals")
    return states, qvals


def make_greedy_pipeline(file_prefix):
    with tf.name_scope("input_pipeline"):
        states_t, qvals_t = make_greedy_readers(file_prefix)
        states_batch, qvals_batch = tf.train.shuffle_batch([states_t, qvals_t], BATCH_SIZE,
                                                           1003 * BATCH_SIZE, 1000*BATCH_SIZE, num_threads=4)
    return states_batch, qvals_batch


if __name__ == "__main__":
    REPLAY_NAME = "seed=42_alpha=0.1"
    log = infra.setup_logging(logfile="q_greedy.log")
    np.random.seed(42)

    started = last_t = time()
    infra.prepare_bbox()

    state_t, q_vals_t = net.make_vars()
    forward_t = net.make_forward_net(state_t)
    loss_t, opt_t = net.make_loss_and_optimiser(state_t, q_vals_t, forward_t)

    states_batch_t, qvals_batch_t = make_greedy_pipeline("../replays/" + REPLAY_NAME)

    log.info("Staring learning from replay {replay}".format(replay=REPLAY_NAME))
    summs = net.make_summaries()

    with tf.Session() as session:
        summary_writer = tf.train.SummaryWriter("logs/" + REPLAY_NAME, graph_def=session.graph_def)
        saver = tf.train.Saver()

        coordinator = tf.train.Coordinator()
        session.run(tf.initialize_all_variables())
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

        try:
            iter = 0
            while True:
                states, qvals = session.run([states_batch_t, qvals_batch_t])
                loss, _ = session.run([loss_t, opt_t], feed_dict={state_t: states, q_vals_t: qvals})
                iter += 1

                if iter % REPORT_ITERS == 0:
                    speed = (BATCH_SIZE * REPORT_ITERS) / (time() - last_t)

                    log.info("Iter {iter}: loss={loss}, time={duration}, states/sec={speed}".format(
                        iter=iter, loss=loss, duration=timedelta(seconds=time()-started),
                        speed=speed
                    ))

                    last_t = time()

                    feed_dict = {
                        summs['loss']: loss
                    }
                    summary_res, = session.run([summs['summary_t']], feed_dict=feed_dict)
                    summary_writer.add_summary(summary_res, iter)
                    summary_writer.flush()

                if iter % SAVE_MODEL_ITERS == 0:
                    saver.save(session, "models/model", global_step=iter)
        finally:
            coordinator.request_stop()
            coordinator.join(threads)