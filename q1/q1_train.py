import sys
sys.path.append("..")

from lib import infra
import numpy as np
import tensorflow as tf

L1_SIZE = 1024

GAMMA = 0.9
LR = 0.1

def random_action(our_state, bbox_state):
    return np.random.randint(0, infra.n_actions, 1)


def smart_action(our_state, bbox_state):
    # calculate best action from current state
    session = our_state['session']
    state_t = our_state['state_place']
    forward_t = our_state['forward_net']

    q_vals, = session.run([forward_t], feed_dict={state_t: [bbox_state]})
    best_action = np.argmax(q_vals)
    return best_action


def learn_q(our_state, prev_state, action, reward, new_state):
    # calculate Q* from:
    # - reward we've got
    # - best Q-value from new state
    session = our_state['session']
    state_t = our_state['state_place']
    q_vals_t = our_state['q_vals_place']
    forward_t = our_state['forward_net']
    loss_t = our_state['loss']
    opt_t = our_state['optimiser']

    # determine estimation of Q for new state
    states = [prev_state, new_state]
    val, = session.run([forward_t], feed_dict={state_t: states})
    prev_vals, new_vals = val
    best_action = np.argmax(new_vals)
    best_q = new_vals[best_action]

    # desired output for action=action from prev_state
    q_star = reward + GAMMA * best_q

    # run optimiser
    desired_q_vals = np.array(prev_vals)
    desired_q_vals[action] = q_star
    # TODO: try to stop passing loss_t and check for speedup
    loss, _ = session.run([loss_t, opt_t], feed_dict={state_t: [prev_state], q_vals_t: [desired_q_vals]})

#    print("loss={loss}".format(loss=loss))


def make_forward_net(state_t):
    """
    Create forward network which maps state into Q-value for all actions
    :param state_t:
    :return:
    """
#    state_mat = tf.expand_dims(state_t, -1)

    with tf.name_scope("L0"):
        w = tf.Variable(tf.random_normal((infra.n_features, L1_SIZE), mean=0.0, stddev=0.1))
        b = tf.Variable(tf.zeros((L1_SIZE,)))
        l0_out = tf.nn.relu(tf.matmul(state_t, w) + b)

    with tf.name_scope("L1"):
        w = tf.Variable(tf.random_normal((L1_SIZE, infra.n_actions), mean=0.0, stddev=0.1))
        b = tf.Variable(tf.zeros((infra.n_actions,)))
        output = tf.matmul(l0_out, w) + b
        output = tf.squeeze(output)

    return output


def make_loss_and_optimiser(state_t, q_vals_t, forward_t):
    with tf.name_scope("Opt"):
        loss_t = tf.nn.l2_loss(forward_t - q_vals_t)
        optimiser = tf.train.AdamOptimizer(learning_rate=LR)
        opt_t = optimiser.minimize(loss_t)

    return loss_t, opt_t



if __name__ == "__main__":
    np.random.seed(42)
    infra.prepare_bbox()

    state_t = tf.placeholder(tf.float32, (None, infra.n_features), name="State")
    forward_t = make_forward_net(state_t)

    q_vals_t = tf.placeholder(tf.float32, (None, infra.n_actions), name="QVals")
    loss_t, opt_t = make_loss_and_optimiser(state_t, q_vals_t, forward_t)

    with tf.Session() as session:
        our_state = {
            'session': session,
            'state_place': state_t,
            'q_vals_place': q_vals_t,
            'forward_net': forward_t,
            'loss': loss_t,
            'optimiser': opt_t,
        }

        init = tf.initialize_all_variables()
        session.run(init)

        saver = tf.train.Saver()
        global_step = 1

        while True:
            infra.prepare_bbox()
            print "%d: Learning round" % global_step
            sys.stdout.flush()

            # Learning step
            infra.bbox_loop(our_state, random_action, learn_q, verbose=False, max_time=1000)
            infra.bbox.finish(verbose=0)

            # Test run
            print "%d: Training round done, perform test run" % global_step
            sys.stdout.flush()
            infra.prepare_bbox()
            infra.bbox_loop(our_state, smart_action, None, verbose=False, max_time=1000)
            infra.bbox.finish(verbose=1)

#            print "%d: save the model" % global_step
#            saver.save(session, "models/model-v1", global_step=global_step)
            global_step += 1

            sys.stdout.flush()
