import sys
sys.path.append("..")

from time import time
from datetime import timedelta
from lib import infra, net, test_bbox, replays
import numpy as np
import tensorflow as tf

STATES_HISTORY = 4
N_STATE = 36
N_ACTIONS = 4

BATCH_SIZE = 100
REPORT_ITERS = 100
SAVE_MODEL_ITERS = 100000


def make_readers(file_prefix):
    """
    Return states and qvals tensors
    :param file_prefix:
    :return:
    """
    FLOAT_SIZE = 4
    states_reader = tf.FixedLengthRecordReader(STATES_HISTORY * N_STATE * FLOAT_SIZE)
    next_states_reader = tf.FixedLengthRecordReader(STATES_HISTORY * N_STATE * FLOAT_SIZE)
    actions_reader = tf.FixedLengthRecordReader(1)
    rewards_reader = tf.FixedLengthRecordReader(FLOAT_SIZE)
    _, states = states_reader.read(tf.train.string_input_producer([file_prefix + ".states"]))
    _, next_states = next_states_reader.read(tf.train.string_input_producer([file_prefix + ".next_states"]))
    _, actions = actions_reader.read(tf.train.string_input_producer([file_prefix + ".actions"]))
    _, rewards = rewards_reader.read(tf.train.string_input_producer([file_prefix + ".rewards"]))

    states = tf.decode_raw(states, tf.float32, name="decode_states")
    states = tf.reshape(states, (STATES_HISTORY * N_STATE, ), name="reshape_states")
    next_states = tf.decode_raw(next_states, tf.float32, name="decode_next_states")
    next_states = tf.reshape(next_states, (STATES_HISTORY * N_STATE, ), name="reshape_next_states")
    actions = tf.decode_raw(actions, tf.int8, name="decode_actions")
    actions = tf.reshape(actions, (1, ), name="reshape_actions")
    actions = tf.to_int32(actions)
    rewards = tf.decode_raw(rewards, tf.float32, name="decode_rewards")
    rewards = tf.reshape(rewards, (1, ), name="reshape_qvals")
    return states, actions, rewards, next_states


def make_pipeline(file_prefixes):
    with tf.name_scope("input_pipeline"):
        tensors = [make_readers(prefix) for prefix in file_prefixes]
        batched_tensors = tf.train.shuffle_batch_join(tensors, BATCH_SIZE,
                                   2000 * BATCH_SIZE, 1000*BATCH_SIZE)
    return batched_tensors


def write_summaries(session, summ, writer, iter_no, feed_batches, **vals):
    feed = {
        summ[name]: value for name, value in vals.iteritems()
    }
    feed.update(feed_batches)
    summ_res, = session.run([summ['summary_t']], feed_dict=feed)
    writer.add_summary(summ_res, iter_no)
    writer.flush()



if __name__ == "__main__":
    LEARNING_RATE = 0.01
    #REPLAY_NAME = "seed=42_alpha=1.0"
    REPLAY_NAME = "t1r2"
    GAMMA = 0.99
    EXTRA = "_lr=%.3f_gamma=%.2f" % (LEARNING_RATE, GAMMA)

    log = infra.setup_logging(logfile="q2_" + REPLAY_NAME + EXTRA + ".log")
    np.random.seed(42)

    started = last_t = time()
    infra.prepare_bbox()

    state_t, action_t, reward_t, next_state_t = net.make_vars_v2(STATES_HISTORY)
    replays = replays.discover_replays("../replays/r1/")

    states_batch_t, actions_batch_t, rewards_batch_t, next_states_batch_t = make_pipeline(replays)

    # make two networks - one is to train, second is periodically cloned from first
    qvals_t = net.make_forward_net_v2(STATES_HISTORY, state_t, is_trainable=True)
    next_qvals_t = net.make_forward_net_v2(STATES_HISTORY, next_state_t, is_trainable=False)

    loss_t = net.make_loss_v2(BATCH_SIZE, GAMMA, qvals_t, action_t, reward_t, next_qvals_t)
    opt_t, optimiser = net.make_opt_v2(loss_t, LEARNING_RATE, decay_every_steps=50000)
    sync_nets_t = net.make_sync_nets_v2()
    summ = net.make_summaries_v2(loss_t, optimiser)

    log.info("Staring learning {name} from replays:".format(name=REPLAY_NAME))
    map(lambda r: log.info("  - {0}".format(r)), replays)
    report_t = time()

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

        summary_writer = tf.train.SummaryWriter("logs/" + REPLAY_NAME + EXTRA, graph_def=session.graph_def)
        saver = tf.train.Saver()
        loss_batch = []

        try:
            iter = 0

            while True:
                if iter % 10000 == 0:
                    log.info("{iter}: Sync nets".format(iter=iter))
                    session.run([sync_nets_t])

                # get data from input pipeline
                states_batch, actions_batch, rewards_batch, next_states_batch = \
                    session.run([states_batch_t, actions_batch_t, rewards_batch_t, next_states_batch_t])

                feed = {
                    state_t: states_batch,
                    action_t: actions_batch,
                    reward_t: rewards_batch,
                    next_state_t: next_states_batch
                }
                loss, _ = session.run([loss_t, opt_t], feed_dict=feed)
                loss_batch.append(loss)

                if iter % REPORT_ITERS == 0 and iter > 0:
                    report_d = time() - report_t
                    speed = (BATCH_SIZE * REPORT_ITERS) / report_d
                    avg_loss = np.median(loss_batch)
                    loss_batch = []
                    log.info("{iter}: loss={loss} in {duration}, speed={speed:.2f} s/sec".format(
                            iter=iter, loss=avg_loss, duration=timedelta(seconds=report_d),
                            speed=speed
                    ))
                    report_t = time()
                    write_summaries(session, summ, summary_writer, iter, feed, loss=avg_loss, speed=speed, score=None)

                if iter % 100000 == 0 and iter > 0:
                    saver.save(session, "models/model", global_step=iter)
                    log.info("{iter}: test model on real bbox".format(iter=iter))
                    t = time()
                    score = test_bbox.test_net(session, STATES_HISTORY, state_t, qvals_t, save_prefix="replays/%d" % (iter/100000))
                    log.info("{iter}: test done in {duration}, score={score}".format(
                        iter=iter, duration=timedelta(seconds=time()-t), score=score
                    ))
                    write_summaries(session, summ, summary_writer, iter, feed, score=score)

                iter += 1
        finally:
            coordinator.request_stop()
            coordinator.join(threads)