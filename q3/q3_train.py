import sys
sys.path.append("..")

from time import time
from datetime import timedelta
from lib import infra, net, replays, features
import numpy as np
import tensorflow as tf

BATCH_SIZE = 1000
REPORT_ITERS = 100
SAVE_MODEL_ITERS = 5000

# if set to None, SYNC_LOSS_THRESHOLD will be used which syncs nets when mean loss for a batch falls below threshold
SYNC_MODELS_ITERS = 5000
SYNC_LOSS_THRESHOLD = 900.0

TEST_CUSTOM_BBOX_ITERS = 0

REPLAY_BUFFER_CAPACITY = 2000000
REPLAY_STEPS_INITIAL = 200000 #400000
REPLAY_STEPS_PER_POLL = 50000
REPLAY_RESET_AFTER_STEPS = 20000

# how many epoches we should show data between fresh replay data requests
EPOCHES_BETWEEN_POLL = 15

DECAY_STEPS = None #200000

# size of queue with fully-prepared train batches. Warning: they eat up a lot of memory!
BATCHES_QUEUE_CAPACITY = 100


def write_summaries(session, summ, writer, iter_no, feed_batches, **vals):
    feed = {
        summ[name]: value for name, value in vals.iteritems()
    }
    feed.update(feed_batches)
    summ_res, = session.run([summ['summary_t']], feed_dict=feed)
    writer.add_summary(summ_res, iter_no)
    writer.flush()


def alpha_from_iter(iter_no):
    if iter < 400000:
        return 1.0
    elif iter <= 2000000:
        return 1.0 - (float(iter_no) / 2000000) + 0.1
    else:
        return 0.1


if __name__ == "__main__":
    LEARNING_RATE = 5e-5
    TEST_NAME = "t32r4"
    TEST_DESCRIPTION = "Sparse data"
    RESTORE_MODEL = None #"models/model_t29r3-100000"
    GAMMA = 0.99
    L2_REG = 0.1

    log = infra.setup_logging(logfile="q3_" + TEST_NAME + ".log")
    np.random.seed(42)

    started = last_t = time()
    infra.prepare_bbox()

    n_features = features.RESULT_N_FEATURES

    # create variables:
    # - state_idx_t, state_val_t -- sparse representation of transformed state
    # - state_t = tf.sparse_to_dense
    # - rewards_t
    state_idx_t, state_val_t, state_t, rewards_t, \
    next_state_idx_t, next_state_val_t, next_state_t = \
        net.make_vars(features.ORIGIN_N_FEATURES, features.RESULT_N_FEATURES, BATCH_SIZE)

    # make two networks - one is to train, second is periodically cloned from first
    qvals_t = net.make_forward_net(state_t, n_features=n_features, is_main_net=True)
    next_qvals_t = net.make_forward_net(next_state_t, n_features=n_features, is_main_net=False)

    # describe qvalues
    tf.contrib.layers.summarize_tensor(tf.reduce_mean(tf.reduce_max(qvals_t, 1), name="qbest"))
    tf.contrib.layers.summarize_tensor(tf.reduce_mean(tf.reduce_max(next_qvals_t, 1), name="qbest_next"))

    loss_t, loss_vec_t = net.make_loss(BATCH_SIZE, GAMMA, qvals_t, rewards_t, next_qvals_t, l2_reg=L2_REG)
    opt_t, optimiser, global_step = net.make_opt(loss_t, LEARNING_RATE, decay_every_steps=DECAY_STEPS)

    sync_nets_t = net.make_sync_nets()
    summ = net.make_summaries(loss_t, optimiser)

    log.info("Staring session {name} with features len {n_features}. Description: {descr}".format(
            name=TEST_NAME, n_features=n_features, descr=TEST_DESCRIPTION))
    report_t = time()

    with tf.Session() as session:
        replay_generator = replays.ReplayGenerator(REPLAY_STEPS_PER_POLL, session, state_t,
                                                   qvals_t, initial=REPLAY_STEPS_INITIAL,
                                                   reset_after_steps=REPLAY_RESET_AFTER_STEPS)
        replay_buffer = replays.ReplayBuffer(REPLAY_BUFFER_CAPACITY, BATCH_SIZE, replay_generator, EPOCHES_BETWEEN_POLL)
        batches_queue, batches_producer_thread = \
            replays.make_batches_queue_and_thread(session, BATCHES_QUEUE_CAPACITY, replay_buffer)
        batches_data_t = batches_queue.dequeue()
        batches_qsize_t = batches_queue.size()

        saver = tf.train.Saver(var_list=dict(net.get_vars(trainable=True)).values(), max_to_keep=200)
        session.run(tf.initialize_all_variables())
        batches_producer_thread.start()

        if RESTORE_MODEL is not None:
            saver.restore(session, RESTORE_MODEL)
            session.run([sync_nets_t])
            print "Global step: {step}".format(step=session.run([global_step]))

        summary_writer = tf.train.SummaryWriter("logs/" + TEST_NAME, session.graph_def)
        loss_batch = []

        iter = 0
        report_d = 0
        syncs = 0
        time_to_sync = False

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

        b_wait = 0.0
        o_wait = 0.0
        q_wait = 0.0

        try:
            while True:
                # first iters we use zero-initialised next_qvals_t
                if time_to_sync:
                    syncs += 1
                    log.info("{iter}: sync nets #{sync}".format(iter=iter, sync=syncs))
                    session.run([sync_nets_t])
                    time_to_sync = False

                # estimate speed before potential refill to prevent confusing numbers
                if iter % REPORT_ITERS == 0 and iter > 0:
                    report_d = time() - report_t
                    speed = (BATCH_SIZE * REPORT_ITERS) / report_d
                    replay_generator.set_alpha(alpha_from_iter(iter))

                # get data from input pipeline
                t1 = time()
                index_batch, states_idx_batch, states_val_batch, rewards_batch, \
                next_states_idx_batch, next_states_val_batch = session.run(batches_data_t)
                b_wait += time()-t1

                feed = {
                    state_idx_t: states_idx_batch,
                    state_val_t: states_val_batch,
                    rewards_t: rewards_batch,
                    next_state_idx_t: next_states_idx_batch,
                    next_state_val_t: next_states_val_batch
                }
                t1 = time()
                loss, loss_vec, _ = session.run([loss_t, loss_vec_t, opt_t], feed_dict=feed)
                o_wait += time() - t1
                loss_batch.append(loss)
                # feed losses back to replay buffer to reflect priority replay
                t1 = time()
                replay_buffer.enqueue_loss_update(index_batch, loss_vec)
                q_wait += time() - t1

                if iter % REPORT_ITERS == 0 and iter > 0:
                    print "Loss update queue size: %d, q_wait = %f, b_wait = %f, o_wait = %f" % (
                        replay_buffer.losses_updates_queue.qsize(), q_wait, b_wait, o_wait)
                    b_wait = o_wait = q_wait = 0.0
                    batches_qsize, = session.run([batches_qsize_t])
                    report_t = time()
                    avg_loss = np.median(loss_batch)
                    if SYNC_MODELS_ITERS is None:
                        time_to_sync = avg_loss < SYNC_LOSS_THRESHOLD
                    else:
                        time_to_sync = iter % SYNC_MODELS_ITERS == 0 and iter > 0
                    loss_batch = []
                    log.info("{iter}: loss={loss:.3f} in {duration}, speed={speed:.2f} s/sec, "
                             "{replay}, batch_q={batches_qsize} ({batchq_perc:.2f}%)".format(
                            iter=iter, loss=avg_loss, duration=timedelta(seconds=report_d),
                            speed=speed, replay=replay_buffer, batches_qsize=batches_qsize,
                            batchq_perc=100.0 * batches_qsize / BATCHES_QUEUE_CAPACITY)
                    )
                    write_summaries(session, summ, summary_writer, iter, feed, loss=avg_loss, speed=speed)

                if TEST_CUSTOM_BBOX_ITERS > 0 and iter % TEST_CUSTOM_BBOX_ITERS == 0 and iter > 0:
                    log.info("{iter} Do custom model states:".format(iter=iter))
                    for state in infra.bbox._all_states():
                        qvals, = session.run([qvals_t], feed_dict={
                            state_t: [state] * BATCH_SIZE
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
