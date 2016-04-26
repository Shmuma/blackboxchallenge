import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    idx_t = tf.placeholder(tf.int64, shape=(4,))
    val_t = tf.placeholder(tf.float32, shape=(4,))

    dense_t = tf.sparse_to_dense(idx_t, (100,), val_t)
    res_t = tf.reduce_sum(dense_t)

    with tf.Session() as session:
        res, dense, = session.run([res_t, dense_t], feed_dict={
            idx_t: [0, 10, 20, 30],
            val_t: [1, 2, 3, 4],
        })

        print dense
        print res
