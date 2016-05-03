import numpy as np
import tensorflow as tf

N_ACTIONS = 4

L1_SIZE = 256
L2_SIZE = 256
L3_SIZE = 128


def leaky_relu(x_t, name, alpha=0.01):
    return tf.maximum(x_t * alpha, x_t)


def make_forward_net(states_t, n_features):
    init = tf.contrib.layers.xavier_initializer()

    with tf.name_scope("L0_T"):
        w = tf.Variable(init((n_features, L1_SIZE)), name="w")
        b = tf.Variable(tf.zeros((L1_SIZE,)), name="b")
        v = tf.matmul(states_t, w, a_is_sparse=True) + b
        l0_out = leaky_relu(v, name="L0")

    with tf.name_scope("L1_T"):
        w = tf.Variable(init((L1_SIZE, L2_SIZE)), name="w")
        b = tf.Variable(tf.zeros((L2_SIZE,)), name="b")
        v = tf.matmul(l0_out, w) + b
        l1_out = tf.nn.relu(v)

    with tf.name_scope("L2_T"):
        w = tf.Variable(init((L2_SIZE, L3_SIZE)), name="w")
        b = tf.Variable(tf.zeros((L3_SIZE,)), name="b")
        v = tf.matmul(l1_out, w) + b
        l2_out = tf.nn.relu(v)

    with tf.name_scope("L3_T"):
        w = tf.Variable(init((L3_SIZE, N_ACTIONS)), name="w")
        b = tf.Variable(tf.zeros((N_ACTIONS,)), name="b")
        output = tf.add(tf.matmul(l2_out, w), b, name="qvals")

    return output



def calc_qvals(network, state):
    """
    Calculate qvalues from extracted network and given bbox state (feature-transformed and converted to dense format)

    :param network:
    :return: qvalues array
    """
    # NB: we don't need to compensate dropout
    LRELU_ALPHA = 0.01
    l0_out = state.dot(network['L0_T/w:0']) + network['L0_T/b:0']
    l0_out = np.maximum(LRELU_ALPHA * l0_out, l0_out)

    l1_out = l0_out.dot(network['L1_T/w:0']) + network['L1_T/b:0']
    l1_out = np.maximum(l1_out, 0.0)

    l2_out = l1_out.dot(network['L2_T/w:0']) + network['L2_T/b:0']
    l2_out = np.maximum(l2_out, 0.0)

    l3_out = l2_out.dot(network['L3_T/w:0']) + network['L3_T/b:0']
    return l3_out


def _get_names():
    res = []
    for layer in range(4):
        for var in ['w', 'b']:
            res.append("L{layer}_T/{var}:0".format(layer=layer, var=var))
    return res


def save_weights(network, file_name):
    data = []
    for name in _get_names():
        data.append(network[name])
    np.save(file_name, data)


def load_weights(file_name):
    data = np.load(file_name)
    res = {}

    for idx, name in enumerate(_get_names()):
        res[name] = data[idx]

    return res