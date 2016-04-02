import sys
sys.path.append("..")

from time import time
from datetime import timedelta
from lib import infra, net, test_bbox
import numpy as np
import tensorflow as tf

STATES_HISTORY = 4
N_STATE = 36
N_ACTIONS = 4

BATCH_SIZE = 100
REPORT_ITERS = 100
SAVE_MODEL_ITERS = 3000


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


def make_pipeline(file_prefix):
    with tf.name_scope("input_pipeline"):
        states_t, actions_t, rewards_t, next_states_t = make_readers(file_prefix)
        states_batch, actions_batch, rewards_batch, next_states_batch = \
            tf.train.shuffle_batch([states_t, actions_t, rewards_t, next_states_t], BATCH_SIZE,
                                   2000 * BATCH_SIZE, 1000*BATCH_SIZE, num_threads=4)
    return states_batch, actions_batch, rewards_batch, next_states_batch


if __name__ == "__main__":
    LEARNING_RATE = 0.1
    REPLAY_NAME = "seed=42_alpha=1.0"
    GAMMA = 0.99
    EXTRA = "_lr=%.3f_gamma=%.2f" % (LEARNING_RATE, GAMMA)

    log = infra.setup_logging(logfile="q2" + EXTRA + ".log")
    np.random.seed(42)

    started = last_t = time()
    infra.prepare_bbox()

    state_t, action_t, reward_t, next_state_t = net.make_vars_v2(STATES_HISTORY)
    states_batch_t, actions_batch_t, rewards_batch_t, next_states_batch_t = make_pipeline("../replays/" + REPLAY_NAME)

    # make two networks - one is to train, second is periodically cloned from first
    qvals_t = net.make_forward_net_v2(STATES_HISTORY, state_t, is_trainable=True)
    next_qvals_t = net.make_forward_net_v2(STATES_HISTORY, next_state_t, is_trainable=False)

    loss_t = net.make_loss_v2(BATCH_SIZE, GAMMA, qvals_t, action_t, reward_t, next_qvals_t)
    opt_t = net.make_opt_v2(loss_t, LEARNING_RATE)
    sync_nets_t = net.make_sync_nets_v2()

    log.info("Staring learning from replay {replay}".format(replay=REPLAY_NAME))

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

        try:
            iter = 0

            while True:
                if iter % 1000 == 0:
                    log.info("{iter}: Sync nets".format(iter=iter))
                    session.run([sync_nets_t])

                if iter % 10000 == 0 and iter > 0:
                    log.info("{iter}: test model on real bbox".format(iter=iter))
                    t = time()
                    score = test_bbox.test_net(session, STATES_HISTORY, state_t, qvals_t)
                    log.info("{iter}: test done in {duration}, score={score}".format(
                        iter=iter, duration=timedelta(seconds=time()-t), score=score
                    ))

                # get data from input pipeline
                states_batch, actions_batch, rewards_batch, next_states_batch = \
                    session.run([states_batch_t, actions_batch_t, rewards_batch_t, next_states_batch_t])

                loss, _ = session.run([loss_t, opt_t], feed_dict={
                    state_t: states_batch,
                    action_t: actions_batch,
                    reward_t: rewards_batch,
                    next_state_t: next_states_batch
                })

                if iter % 100 == 0:
                    log.info("{iter}: loss={loss}".format(iter=iter, loss=loss))

                iter += 1
        finally:
            coordinator.request_stop()
            coordinator.join(threads)