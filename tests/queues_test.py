import tensorflow as tf
import time
import threading


class ProducerQueue(threading.Thread):
    def __init__(self, session, queue_op, qsize_op, size_limit):
        threading.Thread.__init__(self)
        self.session = session
        self.queue_op = queue_op
        self.qsize_op = qsize_op
        self.size_limit = size_limit
        self.stop_requested = False

    def run(self):
        print "Producer: started"
        while not self.stop_requested:
            size, = session.run([self.qsize_op])
            if size <= self.size_limit / 2:
                session.run([self.queue_op])
                print "Producer: produced 1 and 2"
            else:
                print "Producer: size is %d, wait" % size
            time.sleep(1)
        print "Producer: stop requested"


if __name__ == "__main__":
    queue = tf.FIFOQueue(10, (tf.int32,))
    qsize_t = queue.size()
    enqueue_t = queue.enqueue_many([[1, 2]])

    data_t = queue.dequeue()
    with tf.Session() as session:
        producer = ProducerQueue(session, enqueue_t, qsize_t, 10)
        producer.start()

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

        try:
            while True:
                res = session.run([data_t])
                print "Main loop: %s" % str(res)
                time.sleep(1)
        finally:
            producer.stop_requested = True
            coordinator.request_stop()
            coordinator.join(threads)
            producer.join()
