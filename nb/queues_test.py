import tensorflow as tf


if __name__ == "__main__":
    queue = tf.FIFOQueue(10, (tf.int32))
    enqueue_t = queue.enqueue(tf.constant(1))
    qrunner = tf.train.QueueRunner(queue, enqueue_ops=[enqueue_t])
    tf.train.add_queue_runner(qrunner)
    queue_val_t = queue.dequeue()

    print qrunner
    print enqueue_t
    print queue_val_t

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)
        print threads
        size_t = queue.size()

        try:
            for _ in range(100):
                val = session.run([queue_val_t, size_t])
                print val


        finally:
            coordinator.request_stop()
            coordinator.join(threads)