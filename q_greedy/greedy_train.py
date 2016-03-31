import sys
sys.path.append("..")

from lib import infra, net
import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    log = infra.setup_logging()
    np.random.seed(42)

    infra.prepare_bbox()

    state_t, q_vals_t = net.make_vars()
    forward_t = net.make_forward_net(state_t)
    loss_t, opt_t = net.make_loss_and_optimiser(state_t, q_vals_t, forward_t)

    with tf.Session() as session:
        init = tf.initialize_all_variables()
        session.run(init)

        epoch = 0
        while True:

            pass
