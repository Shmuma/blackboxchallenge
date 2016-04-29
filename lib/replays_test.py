import unittest
import numpy as np
import tensorflow as tf

import replays


class TestGenerator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def next_batch(self):
        batch = []
        for i in xrange(self.batch_size):
            state = np.array([i, i*2]), np.array([1.0, 1.0])
            rewards = np.array([0.1]*4)
            next_states = [state]*4
            batch.append((state, rewards, next_states))
        return batch


class ReplayBufferTests(unittest.TestCase):
    def _make_rb(self):
        return replays.ReplayBuffer(tf.Session(), capacity=10, batch=2, replay_generator=TestGenerator(5), epoches_between_pull=2)

    def test_pull_data(self):
        rb = self._make_rb()
        self.assertTrue(rb.time_to_pull())
        rb.pull_more_data()
        self.assertAlmostEqual(rb.probabs.sum(), 1.0, places=5)
        self.assertAlmostEqual(sum(rb.losses), 5.0)
        self.assertEqual(len(rb.buffer), 5)
        self.assertEqual(len(rb.losses), 5)

        rb.pull_more_data()
        self.assertAlmostEqual(rb.probabs.sum(), 1.0, places=5)
        self.assertAlmostEqual(sum(rb.losses), 10.0)
        self.assertEqual(len(rb.buffer), 10)
        self.assertEqual(len(rb.losses), 10)

        rb.pull_more_data()
        self.assertAlmostEqual(rb.probabs.sum(), 1.0, places=5)
        self.assertAlmostEqual(sum(rb.losses), 10.0)
        self.assertEqual(len(rb.buffer), 10)
        self.assertEqual(len(rb.losses), 10)

    def test_batch(self):
        rb = self._make_rb()
        batch = rb.next_batch()
        self.assertEqual(len(batch), 6)
        index, states_idx, states_val, rewards, next_states_idx, next_states_val = batch
        self.assertEqual(len(index), 2)

