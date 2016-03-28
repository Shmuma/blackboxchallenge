import sys
sys.path.append("..")

from lib import infra
import numpy as np
import tensorflow as tf

L1_SIZE = 1024


def random_action(our_state):
    return np.random.randint(0, infra.n_actions, 1)


def learn_q(our_state, prev_state, action, reward, new_state):
    # calculate Q* from:
    # - reward we've got
    # - best Q-value from new state
    session = our_state['session']
    state_t = our_state['state_place']
    forward_t = our_state['forward_net']

    # determine estimation of Q for new state
    val, = session.run([forward_t], feed_dict={state_t: new_state})
    best_action = np.argmax(val)
    best_q = val[best_action]

    if our_state['init']:
        print best_action, best_q, val

#    our_state['init'] = False


def make_forward_net(state_t):
    """
    Create forward network which maps state into Q-value for all actions
    :param state_t:
    :return:
    """
    state_mat = tf.expand_dims(state_t, -1)

    with tf.name_scope("L0"):
        w = tf.Variable(tf.random_normal((infra.n_features, L1_SIZE), mean=0.0, stddev=0.1))
        b = tf.Variable(tf.zeros((L1_SIZE,)))
        l0_out = tf.nn.relu(tf.matmul(state_mat, w, transpose_a=True) + b)

    with tf.name_scope("L1"):
        w = tf.Variable(tf.random_normal((L1_SIZE, infra.n_actions), mean=0.0, stddev=0.1))
        b = tf.Variable(tf.zeros((infra.n_actions,)))
        output = tf.matmul(l0_out, w) + b
        output = tf.squeeze(output)

    return output


if __name__ == "__main__":
    np.random.seed(42)
    infra.prepare_bbox()

    state_t = tf.placeholder(tf.float32, (infra.n_features,), name="State")
    forward_t = make_forward_net(state_t)

    with tf.Session() as session:
        our_state = {
            'session': session,
            'state_place': state_t,
            'forward_net': forward_t,
            'init': True,
        }

        init = tf.initialize_all_variables()
        session.run(init)

        infra.bbox_loop(our_state, random_action, learn_q)

    infra.bbox.finish(verbose=1)
