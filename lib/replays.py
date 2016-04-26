import threading

import tensorflow as tf
import numpy as np
import time
import logging as log

from datetime import timedelta

import features
import infra


class ReplayBuffer:
    def __init__(self, capacity, batch, replay_generator, epoches_between_pull):
        self.capacity = capacity
        self.batch = batch
        self.buffer = []
        self.replay_generator = replay_generator
        self.batches = 0
        self.epoches_between_pull = epoches_between_pull
        self.batches_to_pull = 0
        # priority replay stuff
        self.losses = []                # array with errors for every sample
        self.max_loss = 1.0             # initial value for max_loss
        self.EPS = 1e-5
        self.recalc_probabs = False

    def time_to_pull(self):
        return len(self.buffer) == 0 or self.batches_to_pull <= 0

    def next_batch(self):
        """
        Return next batch of data
        :return:
        """
        if self.time_to_pull():
            self.pull_more_data()
            self.batches_to_pull = self.epoches_between_pull * len(self.buffer) / self.batch

        if self.recalc_probabs:
            self.calc_probabs()
        index = np.random.choice(len(self.buffer), size=self.batch, replace=False, p=self.probabs)

        states = np.zeros((self.batch, features.RESULT_N_FEATURES))
        rewards = []
        next_states = np.zeros((self.batch, 4, features.RESULT_N_FEATURES))

        for batch_ofs, idx in enumerate(index):
            state, reward, next_4_state = self.buffer[idx]
            states[batch_ofs][state[0]] = state[1]
            for action_id, next_state in enumerate(next_4_state):
                next_states[batch_ofs, action_id][next_state[0]] = next_state[1]
            rewards.append(reward)

        self.batches_to_pull -= 1
        self.batches += 1
        return index, states, rewards, next_states

    def pull_more_data(self):
        """
        Populate more data from replay_generator and append it to our buffer. Remove expired entries if neccessary
        :return:
        """
        next_batch = self.replay_generator.next_batch()
        self.buffer += next_batch
        # new entries populated with max_loss to ensure they'll be shown at least once
        self.losses += [self.max_loss] * len(next_batch)
        # TODO: maybe, we can pop values more effectively
        while len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.losses.pop(0)
        self.recalc_probabs = True

    def calc_probabs(self):
        # calculate priorities array
        self.max_loss = max(self.losses)
        self.probabs = np.array(self.losses) + self.EPS
        self.probabs /= self.probabs.sum()
        self.recalc_probabs = False

    def buffer_size(self):
        """
        Return size in bytes of all entries
        :return:
        """
        size = 0
        for state, reward, next_states in self.buffer:
            size += state[0].nbytes + state[1].nbytes
            size += reward.nbytes
            size += sum(map(lambda s: s[0].nbytes + s[1].nbytes, next_states))
        return size

    def __str__(self):
        return "ReplayBuffer[size={size}, to_pull={to_pull}, max_loss={max_loss:.4e}]".format(
            size=len(self.buffer), to_pull=self.batches_to_pull, max_loss=self.max_loss
        )

    def set_losses(self, index, losses):
        for idx, loss in zip(index, losses):
            self.losses[idx] = loss
        if self.batches % 10 == 0:
            self.recalc_probabs = True

class ReplayBatchProducer(threading.Thread):
    def __init__(self, session, capacity, replay_buffer, qsize_t, enqueue_op, vars):
        threading.Thread.__init__(self)
        self.session = session
        self.capacity = capacity
        self.replay_buffer = replay_buffer
        self.qsize_t = qsize_t
        self.enqueue_op = enqueue_op
        self.vars = vars
        self.stop_requested = False

    def run(self):
        log.info("ReplayBatchProducer: started")

        while not self.stop_requested:
            qsize, = self.session.run([self.qsize_t])
            if qsize >= self.capacity-1:
                time.sleep(1)
                continue

            # TODO: prevent memory leak
            if self.replay_buffer.time_to_pull() and qsize > 0:
                time.sleep(1)
                continue

            index, states, rewards, next_states = self.replay_buffer.next_batch()
            feed = {
                self.vars[0]: index,
                self.vars[1]: states,
                self.vars[2]: rewards,
                self.vars[3]: next_states,
            }
            self.session.run([self.enqueue_op], feed_dict=feed)

    def stop(self):
        self.stop_requested = True
        log.info("ReplayBatchProducer: stop requested")


def make_batches_queue_and_thread(session, capacity, replay_buffer):
    """
    Create fifo queue and start production thread
    :param session:
    :param capacity:
    :param replay_buffer:
    :return:
    """
    queue = tf.FIFOQueue(capacity, (tf.int32, tf.float32, tf.float32, tf.float32))
    qsize_t = queue.size()

    # make varibles for data to be placed in the queue
    index_var_t = tf.placeholder(tf.int32)
    states_var_t = tf.placeholder(tf.float32)
    rewards_var_t = tf.placeholder(tf.float32)
    next_states_var_t = tf.placeholder(tf.float32)
    vars = [index_var_t, states_var_t, rewards_var_t, next_states_var_t]
    enqueue_op = queue.enqueue(vars)

    producer_thread = ReplayBatchProducer(session, capacity, replay_buffer,
                                          qsize_t, enqueue_op, vars)
    return queue, producer_thread


class ReplayGenerator:
    """
    Class generates batches of replay data.
    """
    def __init__(self, batch_size, session, states_t, qvals_t, alpha=1.0, initial=None, reset_after_steps=None):
        self.batch_size = batch_size
        self.initial = initial
        self.session = session
        self.states_t = states_t
        self.qvals_t = qvals_t
        self.alpha = alpha
        self.reset_after_steps = reset_after_steps
        self.reset_bbox()

    def reset_bbox(self):
        log.info("ReplayGenerator: bbox resetted at time step %d" % infra.bbox.get_time())
        infra.prepare_bbox()
        self.has_next = True
        self.score = 0.0

    def next_batch(self):
        """
        Generate next batch
        :param alpha:
        :return: tuple of lists (states, rewards, next_states)
        """
        size = self.batch_size
        if self.initial is not None:
            size = self.initial
            self.initial = None
        log.info("ReplayGenerator: generate next batch of %d with alpha=%.3f", size, self.alpha)
        t = time.time()
        batch = []
        while len(batch) < size:
            state = features.transform(infra.bbox.get_state())
            rewards_states = infra.dig_all_actions(self.score)
            rewards, next_states = zip(*rewards_states)
            next_states = map(features.transform, next_states)
            batch.append((state, np.copy(rewards), next_states))

            # find an action to take
            if np.random.random() < self.alpha:
                action = np.random.randint(0, infra.n_actions, 1)[0]
            else:
                qvals, = self.session.run([self.qvals_t], feed_dict={self.states_t: [features.to_dense(state)]})
                action = np.argmax(qvals)

            self.has_next = infra.bbox.do_action(action)
            self.score = infra.bbox.get_score()
            if self.time_to_reset():
                self.reset_bbox()
        log.info("ReplayGenerator: generated in %s", timedelta(seconds=time.time() - t))
        return batch

    def time_to_reset(self):
        if not self.has_next:
            return True
        if self.reset_after_steps is not None:
            if infra.bbox.get_time() > self.reset_after_steps:
                return True
        return False

    def set_alpha(self, alpha):
        self.alpha = alpha