import sys
sys.path.append("..")

from time import time
from datetime import timedelta
from lib import infra, net, replays, features
from lib.reload_opt import OptionLoader
import numpy as np
import tensorflow as tf

BATCH_SIZE = 2000
REPORT_ITERS = 100
SAVE_MODEL_ITERS = 5000

# If we did at least 10000 iterations since last sync or average loss fall below threshold we do sync.
# To avoid unneeded sync after new replay buffer pull, we wait for 1000 iterations after fresh pull
SYNC_MODELS_ITERS = 5000

SYNC_LOSS_THRESHOLD = 300.0
BATCHES_AFTER_PULL_TO_SYNC = 500

TEST_CUSTOM_BBOX_ITERS = 0

REPLAY_BUFFER_CAPACITY = 1000000
REPLAY_STEPS_INITIAL = 1000000
REPLAY_STEPS_PER_POLL = 50000
REPLAY_RESET_AFTER_STEPS = 100000

FIXED_ALPHA = None

# how many epoches we should show data between fresh replay data requests
EPOCHES_BETWEEN_POLL = 10

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
    if FIXED_ALPHA is not None:
        return FIXED_ALPHA
    if iter < 100000:
        return 1.0
    elif iter <= 200000:
        return 1.0 - (float(iter_no) / 100000) + 0.1
    else:
        return 0.1


def check_options(loader, replay_generator, replay_buffer):
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
                        "SYNC_LOSS_THRESHOLD", "FIXED_ALPHA"}:
                globals()[name] = val
                log.info(msg)
            elif name == "REPLAY_RESET_AFTER_STEPS":
                globals()[name] = val
                replay_generator.set_reset_after_steps(val)
                log.info(msg)
            elif name == "EPOCHES_BETWEEN_POLL":
                globals()[name] = val
                replay_buffer.set_epoches_between_poll(val)
                log.info(msg)
            elif name == "REPLAY_STEPS_PER_POLL":
                globals()[name] = val
                log.info(msg)
                replay_generator.set_batch_size(val)
            else:
                log.info("Variable {name} cannot be modified using config, value {value} ignored".format(
                    name=name, value=val
                ))


if __name__ == "__main__":
    LEARNING_RATE = 1e-4
    TEST_NAME = "t39r1"
    TEST_DESCRIPTION = "200k"
    RESTORE_MODEL = "models/model_t38r1-400000"
    GAMMA = 0.99
    L2_REG = 0.1

    log = infra.setup_logging(logfile="q3_" + TEST_NAME + ".log")
    np.random.seed(42)

    started = last_t = time()
    infra.prepare_bbox()

    n_features = features.RESULT_N_FEATURES
    opts_loader = OptionLoader("options.cfg")

    with tf.Session() as session:
        batches_queue = tf.FIFOQueue(BATCHES_QUEUE_CAPACITY, (
            tf.int32,           # batch index -- offsets within replay buffer
            tf.int32,           # state sparse index vector
            tf.float32,         # state sparse values vector
            tf.float32,         # rewards
            tf.int32,           # next state sparse index vector
            tf.float32))        # next state sparse values vector
        batches_qsize_t = batches_queue.size()

        # batch
        (index_batch_t, states_idx_batch_t, states_val_batch_t, rewards_batch_t,
            next_states_idx_batch_t, next_states_val_batch_t) = batches_queue.dequeue()

        # create variables:
        # - state_idx_t, state_val_t -- sparse representation of transformed state
        # - state_t = tf.sparse_to_dense
        # - rewards_t
        state_t = net.sparse_batch_to_dense(states_idx_batch_t, states_val_batch_t, BATCH_SIZE,
                                            features.ORIGIN_N_FEATURES, features.RESULT_N_FEATURES, name="state")
        next_state_t = net.sparse_batch_to_dense(next_states_idx_batch_t, next_states_val_batch_t,
                                                 BATCH_SIZE * infra.n_actions, features.ORIGIN_N_FEATURES,
                                                 features.RESULT_N_FEATURES, name="next")

        # make two networks - one is to train, second is periodically cloned from first
        qvals_t = net.make_forward_net(state_t, n_features=n_features, is_main_net=True)
        next_qvals_t = net.make_forward_net(next_state_t, n_features=n_features, is_main_net=False)

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

        replay_generator = replays.ReplayGenerator(REPLAY_STEPS_PER_POLL, session, state_t,
                                                   qvals_t, initial=REPLAY_STEPS_INITIAL,
                                                   reset_after_steps=REPLAY_RESET_AFTER_STEPS)
        replay_buffer = replays.ReplayBuffer(session, REPLAY_BUFFER_CAPACITY, BATCH_SIZE, replay_generator, EPOCHES_BETWEEN_POLL)
        batches_producer_thread = replays.make_batches_thread(session, batches_queue, BATCHES_QUEUE_CAPACITY, replay_buffer)

        loss_enqueue_t = replay_buffer.losses_updates_queue.enqueue([index_batch_t, loss_vec_t])

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
                    time_to_sync = False
                    iter_last_synced = iter

                # estimate speed before potential refill to prevent confusing numbers
                if iter % REPORT_ITERS == 0 and iter > 0:
                    report_d = time() - report_t
                    speed = (BATCH_SIZE * REPORT_ITERS) / report_d
                    replay_generator.set_alpha(alpha_from_iter(iter))

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
                    check_options(opts_loader, replay_generator, replay_buffer)

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
