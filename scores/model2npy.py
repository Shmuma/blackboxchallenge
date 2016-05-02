import sys
import os
sys.path.append("..")
# comment this to enable GPU TF
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import os
import numpy as np
import tensorflow as tf

from lib import net, net_light, features, infra


DEST_DIR = "npy"

if __name__ == "__main__":
    if not os.path.exists(DEST_DIR):
        print "Make dest dir %s" % DEST_DIR
        os.mkdir(DEST_DIR)

    infra.prepare_bbox()

    n_features = features.transformed_size()
    state_t = tf.placeholder(tf.float32, (None, n_features))
    qvals_t = net.make_forward_net(state_t, True, n_features, dropout_keep_prob=1.0)

    with tf.Session() as session:
        for name in sys.argv[1:]:
            if name.endswith(".meta"):
                continue

            print "Process file %s" % name

            network_weights = net.extract_network(session, name)
            dest_name = os.path.join(DEST_DIR, os.path.basename(name) + ".npy")
            net_light.save_weights(network_weights, dest_name)




