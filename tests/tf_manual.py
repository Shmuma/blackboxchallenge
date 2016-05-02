"""
Load tensorflow model, extract net's weights and do transformations using only numpy
"""
import sys
import os
sys.path.append("..")
# comment this to enable GPU TF
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import numpy as np

from lib import net, net_light, features, infra

MODEL_FILE = "../q3/models/model_t37r4-100000"

if __name__ == "__main__":
    infra.prepare_bbox()

    n_features = features.transformed_size()
    state_t = tf.placeholder(tf.float32, (None, n_features))
    qvals_t = net.make_forward_net(state_t, True, n_features, dropout_keep_prob=1.0)

    with tf.Session() as session:
        network_weights = net.extract_network(session, MODEL_FILE)
        for name, arr in sorted(network_weights.iteritems()):
            print name, arr.shape

    state_sparse = features.transform(np.zeros((features.ORIGIN_N_FEATURES, )))
    state_dense = features.to_dense(state_sparse)

    qvalues = net_light.calc_qvals(network_weights, state_dense)
    print qvalues