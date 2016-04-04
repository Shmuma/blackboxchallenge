import tensorflow as tf


def make_input_pipeline():
    queue = tf.train.range_input_producer(10, shuffle=False)
    value = queue.dequeue()
    print value
    return value


if __name__ == "__main__":
    input_t = make_input_pipeline()

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)
        print threads

        try:
            for _ in xrange(100):
                val, = session.run([input_t])
                print val
        finally:
            coordinator.request_stop()
            coordinator.join(threads)
