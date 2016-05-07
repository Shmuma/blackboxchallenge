import threading

import tensorflow as tf
import numpy as np
import time
import logging as log
from humanize import naturalsize

from datetime import timedelta

import features
import infra
import glob
import os

LOSSES_QUEUE_CAPACITY = 1000


def find_replays(dir):
    """
    Return tuples with file index and file name
    :param dir:
    :return:
    """
    index_files = []
    for f in glob.glob(os.path.join(dir, "replay-*")):
        v = int(f.split("-")[1])
        index_files.append((v, f))
    return index_files


def save_replay_batch(file_name, batch):
    np.save(file_name, batch)


def load_replay_batch(file_name):
    return [tuple(row) for row in np.load(file_name)]


class ReplayBuffer:
    def __init__(self, session, capacity, batch, epoches_between_pull, initial_batches, replays_dir="replays"):
        self.session = session
        self.capacity = capacity
        self.batch = batch
        self.buffer = []
        self.buffer_bytes = None
        self.epoches_between_pull = epoches_between_pull
        # how many batches need to be generated before pull
        self.batches_to_pull = 0
        # amount of batches generated after last pull
        self.batches_since_pull = 0

        # priority replay stuff
        self.max_loss = 1.0               # initial value for max_loss
        # array with errors for every sample
        self.losses = np.full(shape=(0,), fill_value=self.max_loss, dtype=np.float32)
        self.EPS = 1e-5
        self.losses_updates_queue = tf.FIFOQueue(LOSSES_QUEUE_CAPACITY, (tf.int32, tf.float32))
        self.losses_updates_qsize = self.losses_updates_queue.size()
        self.losses_updates_dequeue = self.losses_updates_queue.dequeue()

        self.index = None
        self.index_ofs = 0
        self.replays_dir = replays_dir

        # load initial batches
        self.pull_more_data(initial_batches)

    def time_to_pull(self):
        return len(self.buffer) == 0 or self.batches_to_pull <= 0

    def shuffle(self, pregen_batches=100):
        """
        Generate index of next batch
        :arg pregen_batches: how many batches to generate
        :return: array of batch with entries
        """
        if self.index is None or self.index_ofs >= min(pregen_batches, len(self.buffer) / self.batch):
            self.index = np.random.choice(len(self.buffer), size=min(len(self.buffer), self.batch*pregen_batches),
                                          replace=False, p=self.probabs)
            self.index_ofs = 0
        res = self.index[self.index_ofs*self.batch:(self.index_ofs+1)*self.batch]
        self.index_ofs += 1
        return res

    def next_batch(self):
        """
        Return next batch of data
        :return:
        """
        # apply losses before pulling more data to decrease chance to get outdated loss updates
        self.apply_loss_updates()
        if self.time_to_pull():
            self.pull_more_data()

        index = self.shuffle()

        states_idx = []
        states_val = []
        rewards = []
        next_states_idx = []
        next_states_val = []

        for batch_ofs, idx in enumerate(index):
            state, reward, next_4_state = self.buffer[idx]
            states_idx.append(state[0])
            states_val.append(state[1])
            rewards.append(reward)
            for next_idx, next_val in next_4_state:
                next_states_idx.append(next_idx)
                next_states_val.append(next_val)

        self.batches_to_pull -= 1
        self.batches_since_pull += 1
        return index, states_idx, states_val, rewards, next_states_idx, next_states_val

    def pull_more_data(self, replay_batches=1):
        """
        Populate more data from replay_generator and append it to our buffer. Remove expired entries if neccessary
        :return:
        """
        for _ in range(replay_batches):
            next_batch = self.load_replay_batch()
            self.buffer += next_batch
            # new entries populated with max_loss to ensure they'll be shown at least once
            new_losses = np.full((len(next_batch),), fill_value=self.max_loss, dtype=np.float32)
            self.losses = np.concatenate([self.losses, new_losses])

        if self.losses.shape[0] > self.capacity:
            self.losses = self.losses[-self.capacity:]
            self.buffer = self.buffer[-self.capacity:]

        self.calc_probabs()
        self.buffer_bytes = None
        log.info("Data pulled: {self}".format(self=str(self)))

        self.batches_to_pull = self.epoches_between_pull * len(self.buffer) / self.batch
        self.batches_since_pull = 0
        self.index = None


    def wait_for_replay_file(self):
        """
        Wait for next replay file
        :return: tuple with index and file name
        """
        while True:
            index_files = find_replays(self.replays_dir)
            if len(index_files) == 0:
                log.info("No replay files found, sleep for minute and retry")
                time.sleep(60)
                continue
            index_files.sort()
            return index_files.pop(0)

    def load_replay_batch(self):
        """
        Discover and load batch from external dir. If no batches found, wait for it
        """
        index, file_name = self.wait_for_replay_file()
        log.info("Loading replay batch {file_name}".format(file_name=file_name))
        data = load_replay_batch(file_name)
        os.rename(file_name, os.path.join(self.replays_dir, "done", os.path.basename(file_name)))
        #os.unlink(file_name)
        return data

    def calc_probabs(self):
        # calculate priorities array
        self.max_loss = self.losses.max()
        self.probabs = self.losses + self.EPS
        self.probabs /= self.probabs.sum()

    def buffer_size(self):
        """
        Return size in bytes of all entries
        :return:
        """
        if self.buffer_bytes is not None:
            return self.buffer_bytes

        size = 0
        for state, reward, next_states in self.buffer:
            size += state[0].nbytes + state[1].nbytes
            size += reward.nbytes
            size += sum(map(lambda s: s[0].nbytes + s[1].nbytes, next_states))
        size += self.losses.nbytes
        size += self.probabs.nbytes
        self.buffer_bytes = size
        return size

    def __str__(self):
        return "ReplayBuffer[size={size}({bytes}), to_pull={to_pull}, max_loss={max_loss:.4e}]".format(
                size=len(self.buffer), to_pull=self.batches_to_pull,
                max_loss=self.max_loss, bytes=naturalsize(self.buffer_size(), format="%.3f")
        )

    def apply_loss_updates(self):
        """
        Clean up loss update queue and recalc probabilities
        :return:
        """
        any_changes = False

        while True:
            qsize, = self.session.run([self.losses_updates_qsize])
            if qsize == 0:
                break
            while qsize > 0:
                index, losses = self.session.run(self.losses_updates_dequeue)
                np.put(self.losses, index, losses)
                qsize -= 1
            any_changes = True

        if any_changes:
            self.calc_probabs()

    def set_epoches_between_poll(self, new_epoches):
        self.epoches_between_pull = new_epoches


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

            # TODO: prevent memory leak (is it still there?)
            if self.replay_buffer.time_to_pull() and qsize > 0:
                time.sleep(1)
                continue

            index, states_idx, states_val, rewards, next_states_idx, next_states_val = self.replay_buffer.next_batch()
            feed = dict(zip(self.vars, [index, states_idx, states_val, rewards, next_states_idx, next_states_val]))
            self.session.run([self.enqueue_op], feed_dict=feed)

    def stop(self):
        self.stop_requested = True
        log.info("ReplayBatchProducer: stop requested")


def make_batches_thread(session, queue, capacity, replay_buffer):
    """
    Create fifo queue and start production thread
    :param session:
    :param capacity:
    :param replay_buffer:
    :return:
    """
    qsize_t = queue.size()

    # make varibles for data to be placed in the queue
    index_var_t = tf.placeholder(tf.int32)
    states_idx_t = tf.placeholder(tf.int32)
    states_val_t = tf.placeholder(tf.float32)
    rewards_var_t = tf.placeholder(tf.float32)
    next_states_idx_t = tf.placeholder(tf.int32)
    next_states_val_t = tf.placeholder(tf.float32)
    vars = [index_var_t, states_idx_t, states_val_t, rewards_var_t, next_states_idx_t, next_states_val_t]
    enqueue_op = queue.enqueue(vars)

    producer_thread = ReplayBatchProducer(session, capacity, replay_buffer,
                                          qsize_t, enqueue_op, vars)
    return producer_thread


class ReplayGenerator:
    """
    Class generates batches of replay data.
    """
    def __init__(self, batch_size, session, states_t, qvals_t, alpha=1.0, reset_after_steps=None, cache_actions=None):
        self.batch_size = batch_size
        self.session = session
        self.states_t = states_t
        self.qvals_t = qvals_t
        self.alpha = alpha
        self.reset_after_steps = reset_after_steps
        self.cache_actions = cache_actions
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
        log.info("ReplayGenerator: generate next batch of %d with alpha=%.3f", self.batch_size, self.alpha)
        t = time.time()
        batch = []
        cache_counter = self.cache_actions
        cached_action = None

        while len(batch) < self.batch_size:
            state = features.transform(infra.bbox.get_state())
            rewards_states = infra.dig_all_actions(self.score)
            rewards, next_states = zip(*rewards_states)
            next_states = map(features.transform, next_states)
            batch.append((state, np.copy(rewards), next_states))

            if cache_counter is not None:
                cache_counter -= 1

            # find an action to take
            if np.random.random() < self.alpha:
                action = np.random.randint(0, infra.n_actions, 1)[0]
            else:
                if cache_counter is not None and cache_counter > 0:
                    action = cached_action
                else:
                    qvals, = self.session.run([self.qvals_t], feed_dict={self.states_t: [features.to_dense(state)]})
                    action = np.argmax(qvals)
                    cached_action = action
                    cache_counter = self.cache_actions

            self.has_next = infra.bbox.do_action(action)
            self.score = infra.bbox.get_score()
            if self.time_to_reset() or not self.has_next:
                self.reset_bbox()
            if not self.has_next:
                break
        log.info("ReplayGenerator: generated in %s, bbox_time=%d",
                 timedelta(seconds=time.time() - t), infra.bbox.get_time())
        return batch

    def time_to_reset(self):
        if not self.has_next:
            return True
        if self.reset_after_steps is not None:
            if infra.bbox.get_time() >= self.reset_after_steps:
                return True
        return False

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_reset_after_steps(self, reset):
        self.reset_after_steps = reset

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
