import sys
sys.path.append("..")

from time import time
from datetime import timedelta
from lib import infra, net, run_bbox, replays, features
import numpy as np
import tensorflow as tf

STATES_HISTORY = 1

BATCH_SIZE = 500
REPORT_ITERS = 1000
SAVE_MODEL_ITERS = 100000
SYNC_MODELS_ITERS = 30000
FILL_REPLAY_ITERS = 100000
TEST_PERFORMANCE_ITERS = 50000
TEST_CUSTOM_BBOX_ITERS = 0

# size of queue with fully-prepared train batches. Warning: they eat up a lot of memory!
BATCHES_QUEUE_CAPACITY = 400

REPLAY_STEPS = 400000
#REPLAY_STEPS = None
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
    TEST_NAME = "t25r1"
    TEST_DESCRIPTION = "Full model!"
    RESTORE_MODEL = None #"models-copy/model_t8r1-2000000"
    GAMMA = 0.99
    L2_REG = 0.1

#    infra.init("grid_2x2")
    log = infra.setup_logging(logfile="q3_" + TEST_NAME + ".log")
    np.random.seed(42)

    started = last_t = time()
    infra.prepare_bbox()

    n_features = features.transformed_size()
    replay_buffer = replays.ReplayBuffer(2400000, BATCH_SIZE, STATES_HISTORY)

    state_t, rewards_t, next_state_t = net.make_vars_v3(STATES_HISTORY, n_features)

    # make two networks - one is to train, second is periodically cloned from first
    qvals_t = net.make_forward_net_v3(STATES_HISTORY, state_t, n_features=n_features, is_main_net=True)
    next_qvals_t = net.make_forward_net_v3(STATES_HISTORY, next_state_t, n_features=n_features, is_main_net=False)

    # describe qvalues
    tf.contrib.layers.summarize_tensor(tf.reduce_mean(qvals_t, name="qvals"))
    tf.contrib.layers.summarize_tensor(tf.reduce_mean(next_qvals_t, name="qvals_next"))

    tf.contrib.layers.summarize_tensor(tf.reduce_mean(tf.reduce_max(qvals_t, 1), name="qbest"))
    tf.contrib.layers.summarize_tensor(tf.reduce_mean(tf.reduce_max(next_qvals_t, 1), name="qbest_next"))

    tf.contrib.layers.summarize_tensor(tf.reduce_mean(tf.reduce_min(qvals_t, 1), name="qworst"))
    tf.contrib.layers.summarize_tensor(tf.reduce_mean(tf.reduce_min(next_qvals_t, 1), name="qworst_next"))

    loss_t, qref_t = net.make_loss_v3(BATCH_SIZE, GAMMA, qvals_t, rewards_t, next_qvals_t, l2_reg=L2_REG)
    opt_t, optimiser, global_step = net.make_opt(loss_t, LEARNING_RATE, decay_every_steps=None)

    sync_nets_t = net.make_sync_nets_v2()
    summ = net.make_summaries_v2(loss_t, optimiser)

    log.info("Staring session {name} with features len {n_features}. Description: {descr}".format(
            name=TEST_NAME, n_features=n_features, descr=TEST_DESCRIPTION))
    report_t = time()

    with tf.Session() as session:
        batches_queue, batches_producer_thread = \
            replays.make_batches_queue_and_thread(session, BATCHES_QUEUE_CAPACITY, replay_buffer)
        batches_data_t = batches_queue.dequeue()

        saver = tf.train.Saver(var_list=dict(net.get_v2_vars(trainable=True)).values(), max_to_keep=20)
        session.run(tf.initialize_all_variables())

        if RESTORE_MODEL is not None:
            saver.restore(session, RESTORE_MODEL)

        summary_writer = tf.train.SummaryWriter("logs/" + TEST_NAME, graph_def=session.graph_def)
        loss_batch = []

        iter = 0
        report_d = score_train = score_avg_train = 0
        score_test = score_avg_test = 0

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

        try:
            while True:
                # first iters we use zero-initialised next_qvals_t
                if iter % SYNC_MODELS_ITERS == 0 and iter > 0:
                    log.info("{iter}: sync nets")
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
                    run_bbox.populate_replay_buffer(replay_buffer, session, STATES_HISTORY, state_t, qvals_t,
                                                    alpha=alpha, max_steps=REPLAY_STEPS, verbose=100000)
                    replay_buffer.reshuffle()
                    log.info("{iter}: population done in {duration}".format(
                        iter=iter, duration=timedelta(seconds=time()-t)
                    ))

                if iter % TEST_PERFORMANCE_ITERS == 0 and iter > 0:
                    log.info("{iter}: test performance on train and test levels".format(iter=iter))
                    t = time()
                    score_train, score_avg_train = run_bbox.test_performance(session, STATES_HISTORY, state_t,
                                                                             qvals_t, alpha=0.0, max_steps=REPLAY_STEPS, test_level=False)
                    score_test, score_avg_test = run_bbox.test_performance(session, STATES_HISTORY, state_t,
                                                                           qvals_t, alpha=0.0, max_steps=REPLAY_STEPS, test_level=True)
                    replay_buffer.reshuffle()
                    log.info("{iter}: test done in {duration}, score_train={score_train}, avg_train={score_avg_train:.3e}, "
                             "score_test={score_test}, avg_test={score_avg_test:.3e}".format(
                             iter=iter, duration=timedelta(seconds=time()-t), score_train=score_train,
                             score_avg_train=score_avg_train, score_test=score_test, score_avg_test=score_avg_test
                    ))

                # get data from input pipeline
                #states_batch, rewards_batch, next_states_batch = replay_buffer.next_batch()
                states_batch, rewards_batch, next_states_batch = session.run(batches_data_t)

                feed = {
                    state_t: states_batch,
                    rewards_t: rewards_batch,
                    next_state_t: next_states_batch
                }
                loss, qvals, next_qvals, qref, _ = session.run([loss_t, qvals_t, next_qvals_t, qref_t, opt_t], feed_dict=feed)
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


                if TEST_CUSTOM_BBOX_ITERS > 0 and iter % TEST_CUSTOM_BBOX_ITERS == 0 and iter > 0:
                    log.info("{iter} Do custom model states:".format(iter=iter))
                    for state in infra.bbox._all_states():
                        qvals, = session.run([qvals_t], feed_dict={
                            state_t: [[state]] * BATCH_SIZE
                        })
                        log.info("   {state}: {qvals}".format(
                                state=infra.bbox._describe_state(state),
                                qvals=", ".join(map(lambda v: "%7.3f" % v, qvals[0]))
                        ))

                if iter % SAVE_MODEL_ITERS == 0 and iter > 0:
                    saver.save(session, "models/model_" + TEST_NAME, global_step=iter)

                iter += 1
        finally:
            batches_producer_thread.stop()
            coordinator.request_stop()
            coordinator.join(threads)
            batches_producer_thread.join()