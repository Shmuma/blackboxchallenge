import sys
sys.path.append("..")

from time import time
from datetime import timedelta
from lib import infra, net, test_bbox, replays
import numpy as np
import tensorflow as tf

STATES_HISTORY = 20
N_STATE = 36
N_ACTIONS = 4

BATCH_SIZE = 500
REPORT_ITERS = 1000
SAVE_MODEL_ITERS = 100000
SYNC_MODELS_ITERS = 10000
FILL_REPLAY_ITERS = 2000
TEST_PERFORMANCE_ITERS = 10000

REPLAY_STEPS = 20000

def write_summaries(session, summ, writer, iter_no, feed_batches, **vals):
    feed = {
        summ[name]: value for name, value in vals.iteritems()
    }
    feed.update(feed_batches)
    summ_res, = session.run([summ['summary_t']], feed_dict=feed)
    writer.add_summary(summ_res, iter_no)
    writer.flush()



if __name__ == "__main__":
    LEARNING_RATE = 1e-4
    TEST_NAME = "t12r1"
    RESTORE_MODEL = None #"models-copy/model_t8r1-2000000"
    GAMMA = 0.99
    L2_REG = 0.01

    log = infra.setup_logging(logfile="q3_" + TEST_NAME + ".log")
    np.random.seed(42)

    started = last_t = time()
    infra.prepare_bbox()

    replay_buffer = replays.ReplayBuffer(1000000, BATCH_SIZE)

    state_t, rewards_t, next_state_t = net.make_vars_v3(STATES_HISTORY)

    # make two networks - one is to train, second is periodically cloned from first
    qvals_t = net.make_forward_net_v3(STATES_HISTORY, state_t, is_trainable=True)
    next_qvals_t = net.make_forward_net_v3(STATES_HISTORY, next_state_t, is_trainable=False)

    tf.contrib.layers.summarize_tensor(tf.reduce_mean(qvals_t, name="qvals"))
    tf.contrib.layers.summarize_tensor(tf.reduce_mean(next_qvals_t, name="qvals_next"))

    loss_t = net.make_loss_v3(BATCH_SIZE, GAMMA, qvals_t, rewards_t, next_qvals_t, l2_reg=L2_REG)
    opt_t, optimiser, global_step = net.make_opt(loss_t, LEARNING_RATE, decay_every_steps=100000)
    sync_nets_t = net.make_sync_nets_v2()
    summ = net.make_summaries_v2(loss_t, optimiser)

    log.info("Staring session {name}".format(name=TEST_NAME))
    report_t = time()

    with tf.Session() as session:
        saver = tf.train.Saver(var_list=dict(net.get_v2_vars(trainable=True)).values(), max_to_keep=20)
        session.run(tf.initialize_all_variables())

        if RESTORE_MODEL is not None:
            saver.restore(session, RESTORE_MODEL)

        summary_writer = tf.train.SummaryWriter("logs/" + TEST_NAME, graph_def=session.graph_def)
        loss_batch = []

        iter = 0
        report_d = score_train = score_avg_train = 0
        score_test = score_avg_test = 0

        while True:
            if iter % SYNC_MODELS_ITERS == 0:
                session.run([sync_nets_t])

            # estimate speed before potential refill to prevent confusing numbers
            if iter % REPORT_ITERS == 0 and iter > 0:
                report_d = time() - report_t
                speed = (BATCH_SIZE * REPORT_ITERS) / report_d

            if iter % FILL_REPLAY_ITERS == 0:
                if iter < 200000:
                    alpha = 1.0
                elif iter <= 1000000:
                    alpha = 1.0 - (float(iter) / 1000000) + 0.1
                else:
                    alpha = 0.1

                log.info("{iter}: populating replay buffer with alpha={alpha}".format(
                        iter=iter, alpha=alpha))
                t = time()
                test_bbox.populate_replay_buffer(replay_buffer, session, STATES_HISTORY, state_t, qvals_t,
                                                                    alpha=alpha, max_steps=REPLAY_STEPS)
                replay_buffer.reshuffle()
                log.info("{iter}: population done in {duration}".format(
                    iter=iter, duration=timedelta(seconds=time()-t)
                ))

            if iter % TEST_PERFORMANCE_ITERS == 0 and iter > 0:
                log.info("{iter}: test performance on train and test levels".format(iter=iter))
                t = time()
                score_train, score_avg_train = test_bbox.test_performance(session, STATES_HISTORY, state_t,
                                                              qvals_t, alpha=0.0, max_steps=REPLAY_STEPS, test_level=False)
                score_test, score_avg_test = test_bbox.test_performance(session, STATES_HISTORY, state_t,
                                                              qvals_t, alpha=0.0, max_steps=REPLAY_STEPS, test_level=True)
                replay_buffer.reshuffle()
                log.info("{iter}: test done in {duration}, score_train={score_train}, avg_train={score_avg_train:.3e}, "
                         "score_test={score_test}, avg_test={score_avg_test:.3e}".format(
                         iter=iter, duration=timedelta(seconds=time()-t), score_train=score_train,
                         score_avg_train=score_avg_train, score_test=score_test, score_avg_test=score_avg_test
                ))

            # get data from input pipeline
            states_batch, rewards_batch, next_states_batch = replay_buffer.next_batch()

            feed = {
                state_t: states_batch,
                rewards_t: rewards_batch,
                next_state_t: next_states_batch
            }
            loss, _ = session.run([loss_t, opt_t], feed_dict=feed)
            loss_batch.append(loss)

            if iter % REPORT_ITERS == 0 and iter > 0:
                report_t = time()
                avg_loss = np.median(loss_batch)
                loss_batch = []
                log.info("{iter}: loss={loss} in {duration}, speed={speed:.2f} s/sec, replay={replay}".format(
                        iter=iter, loss=avg_loss, duration=timedelta(seconds=report_d),
                        speed=speed, replay=replay_buffer
                ))
                write_summaries(session, summ, summary_writer, iter, feed, loss=avg_loss, speed=speed,
                                score_train=score_train, score_avg_train=score_avg_train,
                                score_test=score_test, score_avg_test=score_avg_test)

            if iter % SAVE_MODEL_ITERS == 0 and iter > 0:
                saver.save(session, "models/model_" + TEST_NAME, global_step=iter)

            iter += 1
