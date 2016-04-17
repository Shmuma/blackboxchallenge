import array, struct
import glob
import threading

import tensorflow as tf
import numpy as np
import time
import logging as log

import features

def pack_item(item):
    bbox_state, action, reward = item
    return array.array('f', bbox_state.tolist()).tostring(), action, reward


def pack_state(state):
    return array.array('f', state.tolist())


class ReplayWriter:
    def __init__(self, file_prefix):
        self.states_fd = open(file_prefix + ".states", "w+")
        self.actions_fd = open(file_prefix + ".actions", "w+")
        self.rewards_fd = open(file_prefix + ".rewards", "w+")
        self.next_states_fd = open(file_prefix + ".next_states", "w+")

    def close(self):
        self.states_fd.close()
        self.actions_fd.close()
        self.rewards_fd.close()
        self.next_states_fd.close()

    def append(self, state, action, rewards, next_states):
        pack_state(state).tofile(self.states_fd)
        self.actions_fd.write(struct.pack('b', action))
        self.rewards_fd.write(struct.pack('ffff', *rewards))
        for next_state in next_states:
            pack_state(next_state).tofile(self.next_states_fd)

    def append_small(self, states, action, reward, next_states):
        for s in states:
            pack_state(s).tofile(self.states_fd)
        self.actions_fd.write(struct.pack('b', action))
        self.rewards_fd.write(struct.pack('f', reward))
        for s in next_states:
            pack_state(s).tofile(self.next_states_fd)


def discover_replays(path):
    res = []
    suffix = ".states"
    for p in glob.glob(path + "/*" + suffix):
        res.append(p[:-len(suffix)])
    return res


class ReplayBuffer:
    def __init__(self, capacity, batch, history):
        self.capacity = capacity
        self.batch = batch
        self.buffer = []
        self.epoches = 0
        self.history = history
        self.ready_to_generate = False

    def append(self, state, rewards, next_states):
        # prevent consumers from accessing our buffer
        # if state_history == 1, we don't need to do anything
        if self.history == 1:
            tr_state = features.transform(state[0])
            tr_next_states = map(features.transform, next_states)
            self.buffer.append(([tr_state], np.copy(rewards), tr_next_states))
        else:
            st = np.copy(state)
            # combine state history and next_4_state into full next states history
            top_cur_state = st[:-1, :]
            v = []
            for next_state in next_states:
                v.append([np.vstack([next_state, top_cur_state])])
            next_st = np.concatenate(v)

            # transform features
            st = np.apply_along_axis(features.transform, 1, st)
            next_st = np.apply_along_axis(features.transform, 2, next_st)

            self.buffer.append((st, np.copy(rewards), next_st))
        self.ready_to_generate = len(self.buffer) > self.batch * 10

    def reshuffle(self):
        """
        Regenerate shuffled batch. Should be called after batch of appends.
        """
        self.ready_to_generate = False
        while len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        self.shuffle = np.random.permutation(len(self.buffer))
        self.batch_idx = 0
        self.ready_to_generate = True

    def next_batch(self):
        """
        Return next batch of data
        :return:
        """
        if (self.batch_idx + 1) * self.batch > len(self.buffer):
            self.reshuffle()
            self.epoches += 1

        states = np.zeros((self.batch, self.history, features.RESULT_N_FEATURES))
        rewards = []
        next_states = np.zeros((self.batch, 4, self.history, features.RESULT_N_FEATURES))

        for batch_ofs, idx in enumerate(self.shuffle[self.batch_idx*self.batch:(self.batch_idx+1)*self.batch]):
            state, reward, next_4_state = self.buffer[idx]
            if self.history == 1:
                features.apply_dense(states[batch_ofs, 0], state[0])
                for action_id, next_state in enumerate(next_4_state):
                    features.apply_dense(next_states[batch_ofs, action_id, 0], next_state)
            else:
                assert False
            rewards.append(reward)

        self.batch_idx += 1
        return states, rewards, next_states

    def __str__(self):
        return "ReplayBuffer: size={size}, batch={batch}, epoch={epoch}".format(
                size=len(self.buffer), batch=self.batch_idx, epoch=self.epoches)


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
            if not self.replay_buffer.ready_to_generate:
                time.sleep(1)
                continue

            qsize, = self.session.run([self.qsize_t])
            if qsize > self.capacity / 2:
                time.sleep(1)
                continue

            states, rewards, next_states = self.replay_buffer.next_batch()
            feed = {
                self.vars[0]: states,
                self.vars[1]: rewards,
                self.vars[2]: next_states,
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
    queue = tf.FIFOQueue(capacity, (tf.float32, tf.float32, tf.float32))
    qsize_t = queue.size()

    # make varibles for data to be placed in the queue
    states_var_t = tf.placeholder(tf.float32)
    rewards_var_t = tf.placeholder(tf.float32)
    next_states_var_t = tf.placeholder(tf.float32)
    enqueue_op = queue.enqueue([states_var_t, rewards_var_t, next_states_var_t])

    producer_thread = ReplayBatchProducer(session, capacity, replay_buffer,
                                          qsize_t, enqueue_op, (states_var_t, rewards_var_t, next_states_var_t))
    producer_thread.start()

    return queue, producer_thread