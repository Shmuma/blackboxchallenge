import sys
sys.path.append("..")

from time import time
from datetime import timedelta
from lib import infra, net, test_bbox, replays
import numpy as np
import tensorflow as tf

STATES_HISTORY = 10
N_STATE = 36
N_ACTIONS = 4

BATCH_SIZE = 100
REPORT_ITERS = 1000
SAVE_MODEL_ITERS = 100000
SYNC_MODELS_ITERS = 1000
FILL_REPLAY_ITERS = 50000


def write_summaries(session, summ, writer, iter_no, feed_batches, **vals):
    feed = {
        summ[name]: value for name, value in vals.iteritems()
    }
    feed.update(feed_batches)
    summ_res, = session.run([summ['summary_t']], feed_dict=feed)
    writer.add_summary(summ_res, iter_no)
    writer.flush()



if __name__ == "__main__":
    LEARNING_RATE = 1.0e-3
    TEST_NAME = "t5r1"
    RESTORE_MODEL = None #"models/modelt1r1-700000"
    GAMMA = 0.99
    EXTRA = "_lr=%.3f_gamma=%.2f" % (LEARNING_RATE, GAMMA)

    log = infra.setup_logging(logfile="q3_" + TEST_NAME + EXTRA + ".log")
    np.random.seed(42)

    started = last_t = time()
    infra.prepare_bbox()

    replay_buffer = replays.ReplayBuffer(200000, BATCH_SIZE)

    state_t, action_t, reward_t, next_state_t = net.make_vars_v3(STATES_HISTORY)

    # make two networks - one is to train, second is periodically cloned from first
    qvals_t = net.make_forward_net_v3(STATES_HISTORY, state_t, is_trainable=True)
    next_qvals_t = net.make_forward_net_v3(STATES_HISTORY, next_state_t, is_trainable=False)

    loss_t = net.make_loss_v2(BATCH_SIZE, GAMMA, qvals_t, action_t, reward_t, next_qvals_t)
    opt_t, optimiser, global_step = net.make_opt_v2(loss_t, LEARNING_RATE, decay_every_steps=100000)
    sync_nets_t = net.make_sync_nets_v2()
    summ = net.make_summaries_v2(loss_t, optimiser)

    log.info("Staring session {name}{extra}".format(name=TEST_NAME, extra=EXTRA))
    report_t = time()

    with tf.Session() as session:
        saver = tf.train.Saver(var_list=dict(net.get_v2_vars(trainable=True)).values())
        session.run(tf.initialize_all_variables())

        if RESTORE_MODEL is not None:
            saver.restore(session, RESTORE_MODEL)
            print session.run([global_step])
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

        summary_writer = tf.train.SummaryWriter("logs/" + TEST_NAME + EXTRA, graph_def=session.graph_def)
        loss_batch = []

        try:
            iter = 0

            while True:
                if iter % SYNC_MODELS_ITERS == 0:
#                    log.info("{iter}: Sync nets".format(iter=iter))
                    session.run([sync_nets_t])

                if iter % FILL_REPLAY_ITERS == 0:
                    log.info("{iter}: populating replay buffer".format(iter=iter))
                    t = time()
                    score = test_bbox.populate_replay_buffer(replay_buffer, session, STATES_HISTORY, state_t, qvals_t,
                                                             alpha=0.1, max_steps=50000)
                    replay_buffer.reshuffle()
                    log.info("{iter}: test done in {duration}, score={score}".format(
                        iter=iter, duration=timedelta(seconds=time()-t), score=score
                    ))

                # get data from input pipeline
                states_batch, actions_batch, rewards_batch, next_states_batch = replay_buffer.next_batch()

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
                    log.info("{iter}: loss={loss} in {duration}, speed={speed:.2f} s/sec, replay={replay}".format(
                            iter=iter, loss=avg_loss, duration=timedelta(seconds=report_d),
                            speed=speed, replay=replay_buffer
                    ))
                    report_t = time()
                    write_summaries(session, summ, summary_writer, iter, feed, loss=avg_loss, speed=speed, score=score)

                if iter % SAVE_MODEL_ITERS == 0 and iter > 0:
                    saver.save(session, "models/model_" + TEST_NAME, global_step=iter)

                iter += 1
        finally:
            coordinator.request_stop()
            coordinator.join(threads)
