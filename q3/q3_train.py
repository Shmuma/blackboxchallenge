import os
import sys
sys.path.append("..")

from time import time
from datetime import timedelta
from lib import infra, net, replays, features
from lib.reload_opt import OptionLoader
import numpy as np
import tensorflow as tf

BATCH_SIZE = 5000
REPORT_ITERS = 100
SAVE_MODEL_ITERS = 5000
SAVE_MODEL_FOR_REPLAYS = 1000

# If we did at least 10000 iterations since last sync or average loss fall below threshold we do sync.
# To avoid unneeded sync after new replay buffer pull, we wait for 1000 iterations after fresh pull
SYNC_MODELS_ITERS = 10000

SYNC_LOSS_THRESHOLD = 300.0
BATCHES_AFTER_PULL_TO_SYNC = 500

TEST_CUSTOM_BBOX_ITERS = 0

REPLAY_BUFFER_CAPACITY = 2500000
# every replay batch is 50k steps
INITIAL_REPLAY_BATCHES = 20

# how many epoches we should show data between fresh replay data requests
EPOCHES_BETWEEN_POLL = 3

DECAY_STEPS = None #50000

# size of queue with fully-prepared train batches. Warning: they eat up a lot of memory!
BATCHES_QUEUE_CAPACITY = 10

REPLAY_MODELS_DIR = "replays/models"


def write_summaries(session, summ, writer, iter_no, feed_batches, **vals):
    feed = {
        summ[name]: value for name, value in vals.iteritems()
    }
    feed.update(feed_batches)
    summ_res, = session.run([summ['summary_t']], feed_dict=feed)
    writer.add_summary(summ_res, iter_no)
    writer.flush()


def check_options(loader, replay_buffer):
    if loader.check():
        for name, val in loader.values.iteritems():
            if not name in globals():
                log.warn("Unknown variable {name}, value {value}, ignored".format(name=name, value=val))
                continue
            old_val = globals()[name]
            if old_val == val:
                continue

            msg = "  {name}: {old_val} => {new_val}".format(name=name, old_val=old_val, new_val=val)

            if name in {"SYNC_MODELS_ITERS",
                        "SYNC_LOSS_THRESHOLD"}:
                globals()[name] = val
                log.info(msg)
            elif name == "EPOCHES_BETWEEN_POLL":
                globals()[name] = val
                replay_buffer.set_epoches_between_poll(val)
                log.info(msg)
            else:
                log.info("Variable {name} cannot be modified using config, value {value} ignored".format(
                    name=name, value=val
                ))


if __name__ == "__main__":
    LEARNING_RATE = 1e-2
    TEST_NAME = "t53r2"
    TEST_DESCRIPTION = "Reward history + actions history + less features"
    RESTORE_MODEL = "models/model_t53r1-215000"
    GAMMA = 0.99
    L2_REG = 0.3

    log = infra.setup_logging(logfile="q3_" + TEST_NAME + ".log")
    np.random.seed(42)

    started = last_t = time()
    infra.prepare_bbox()

    n_features = features.RESULT_N_FEATURES
    opts_loader = OptionLoader("options.cfg")

    if not os.path.exists(REPLAY_MODELS_DIR):
        os.makedirs(REPLAY_MODELS_DIR)

    with tf.Session() as session:
        batches_queue = tf.FIFOQueue(BATCHES_QUEUE_CAPACITY, (
            tf.int32,           # batch index -- offsets within replay buffer
            tf.float32,            # state values vector
            tf.float32,         # rewards
            tf.float32))           # next state values vector
        batches_qsize_t = batches_queue.size()

        # batch
        (index_batch_t, states_batch_t, rewards_batch_t, next_states_batch_t) = batches_queue.dequeue()

        # make two networks - one is to train, second is periodically cloned from first
        qvals_t = net.make_forward_net(states_batch_t, n_features=n_features, is_main_net=True)
        next_qvals_t = net.make_forward_net(next_states_batch_t, n_features=n_features, is_main_net=False)

        # describe qvalues
        tf.contrib.layers.summarize_tensor(tf.reduce_mean(tf.reduce_max(qvals_t, 1), name="qbest"))
        tf.contrib.layers.summarize_tensor(tf.reduce_mean(tf.reduce_max(next_qvals_t, 1), name="qbest_next"))

        loss_t, loss_vec_t = net.make_loss(BATCH_SIZE, GAMMA, qvals_t, rewards_batch_t, next_qvals_t, l2_reg=L2_REG)
        opt_t, optimiser, global_step = net.make_opt(loss_t, LEARNING_RATE, decay_every_steps=DECAY_STEPS)

        sync_nets_t = net.make_sync_nets()
        summ = net.make_summaries(loss_t, optimiser)

        log.info("Staring session {name} with features len {n_features}. Description: {descr}".format(
                name=TEST_NAME, n_features=n_features, descr=TEST_DESCRIPTION))
        report_t = time()

        replay_buffer = replays.ReplayBuffer(session, REPLAY_BUFFER_CAPACITY, BATCH_SIZE,
                                             EPOCHES_BETWEEN_POLL, INITIAL_REPLAY_BATCHES)
        batches_producer_thread = replays.make_batches_thread(session, batches_queue, BATCHES_QUEUE_CAPACITY, replay_buffer)

        loss_enqueue_t = replay_buffer.losses_updates_queue.enqueue([index_batch_t, loss_vec_t])

        saver = tf.train.Saver(var_list=dict(net.get_vars(trainable=True)).values(), max_to_keep=200)
        # we have a special saver for replays generator
        saver_replays = tf.train.Saver(var_list=dict(net.get_vars(trainable=True)).values(), max_to_keep=3)
        session.run(tf.initialize_all_variables())
        batches_producer_thread.start()

        if RESTORE_MODEL is not None:
            log.info("Loaded model from {file}".format(file=RESTORE_MODEL))
            saver.restore(session, RESTORE_MODEL)
            session.run([sync_nets_t])
            print "Global step: {step}".format(step=session.run([global_step]))

        summary_writer = tf.train.SummaryWriter("logs/" + TEST_NAME, session.graph_def)
        loss_batch = []

        iter = 0
        report_d = 0
        syncs = 0
        iter_last_synced = 0
        time_to_sync = False

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

        try:
            while True:
                # first iters we use zero-initialised next_qvals_t
                if time_to_sync:
                    syncs += 1
                    log.info("{iter}: sync nets #{sync}".format(iter=iter, sync=syncs))
                    session.run([sync_nets_t])
                    replay_buffer.sync_done()
                    time_to_sync = False
                    iter_last_synced = iter

                # estimate speed before potential refill to prevent confusing numbers
                if iter % REPORT_ITERS == 0 and iter > 0:
                    report_d = time() - report_t
                    speed = (BATCH_SIZE * REPORT_ITERS) / report_d

                loss, _, _ = session.run([loss_t, loss_enqueue_t, opt_t])
                loss_batch.append(loss)

                if iter % REPORT_ITERS == 0 and iter > 0:
                    batches_qsize, = session.run([batches_qsize_t])
                    report_t = time()
                    avg_loss = np.median(loss_batch)

                    # decide about sync time
                    if replay_buffer.batches_since_pull >= BATCHES_AFTER_PULL_TO_SYNC:
                        if avg_loss < SYNC_LOSS_THRESHOLD or iter - iter_last_synced > SYNC_MODELS_ITERS:
                            time_to_sync = True
                    loss_batch = []
                    log.info("{iter}: loss={loss:.3f} in {duration}, speed={speed:.2f} s/sec, "
                             "{replay}, batch_q={batches_qsize} ({batchq_perc:.2f}%)".format(
                            iter=iter, loss=avg_loss, duration=timedelta(seconds=report_d),
                            speed=speed, replay=replay_buffer, batches_qsize=batches_qsize,
                            batchq_perc=100.0 * batches_qsize / BATCHES_QUEUE_CAPACITY)
                    )
                    write_summaries(session, summ, summary_writer, iter, {}, loss=avg_loss, speed=speed)
                    check_options(opts_loader, replay_buffer)

                if TEST_CUSTOM_BBOX_ITERS > 0 and iter % TEST_CUSTOM_BBOX_ITERS == 0 and iter > 0:
                    log.info("{iter} Do custom model states:".format(iter=iter))
                    for state in infra.bbox._all_states():
                        qvals, = session.run([qvals_t], feed_dict={
                            states_batch_t: [state] * BATCH_SIZE
                        })
                        log.info("   {state}: {qvals}".format(
                                state=infra.bbox._describe_state(state),
                                qvals=", ".join(map(lambda v: "%7.3f" % v, qvals[0]))
                        ))

                if iter % SAVE_MODEL_ITERS == 0 and iter > 0:
                    saver.save(session, "models/model_" + TEST_NAME, global_step=iter)

                if iter % SAVE_MODEL_FOR_REPLAYS == 0 and iter > 0:
                    saver_replays.save(session, os.path.join(REPLAY_MODELS_DIR, "model_" + TEST_NAME), global_step=iter)

                iter += 1
        finally:
            batches_producer_thread.stop()
            coordinator.request_stop()
            coordinator.join(threads)
            batches_producer_thread.join()
