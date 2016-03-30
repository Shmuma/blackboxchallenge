import sys
sys.path.append("..")

import math
from lib import infra
import numpy as np
import tensorflow as tf

L1_SIZE = 512
L2_SIZE = 512
L3_SIZE = 512
BATCH_SIZE = 100

DO_TEST = False

# discount factor
GAMMA = 0.9

# exploration/exploitation factor
ALPHA = 0.5
# Learning rate
LR = 0.1


def get_q_vals(our_state, bbox_state):
    session = our_state['session']
    state_t = our_state['state_place']
    forward_t = our_state['forward_net']

    q_vals, = session.run([forward_t], feed_dict={state_t: [bbox_state]})
    return q_vals


def random_action(our_state, bbox_state):
    return np.random.randint(0, infra.n_actions, 1)[0]


def action_hook(our_state, bbox_state):
    q_vals = get_q_vals(our_state, bbox_state)

    # according to current gamma, chose random action
    if np.random.random() < our_state['alpha']:
        action = random_action(our_state, bbox_state)
    else:
        action = np.argmax(q_vals)

    our_state['batch'].append((bbox_state, action, q_vals))
    return action


def reward_hook(our_state, reward, last_round):
    our_state['reward'].append(reward)

    if not last_round and len(our_state['batch']) <= BATCH_SIZE:
        return

    batch_states = our_state['batch']
    batch_reward = our_state['reward'][:-1]     # stripping last entry as it doesn't have next state yet

    # do q-learning stuff. In 'batch' entry we have triples with (state, action, q_vals).
    # zip reward to them and create input and desired output for network for learning
    input_arr = []
    output_arr = []
    for idx, ((bbox_state, action, q_vals), reward) in enumerate(zip(batch_states, batch_reward)):
        input_arr.append(bbox_state)

        best_next_q = max(batch_states[idx+1][2])
        # Bellman's equation
        q_vals[action] = reward + our_state['gamma'] * best_next_q
        output_arr.append(q_vals)

    # learn step
    session = our_state['session']
    state_t = our_state['state_place']
    q_vals_t = our_state['q_vals_place']
    opt_t = our_state['optimiser']
    loss_t = our_state['loss']

    loss, _ = session.run([loss_t, opt_t], feed_dict={state_t: input_arr, q_vals_t: output_arr})

    # periodically save summary
    step = infra.bbox.get_time()
    if (step-1) % 1000 == 0:
        summs = our_state['summaries']
        feed_dict = {
            summs['loss']: math.log10(max(loss, 0.000001)),
            summs['score']: infra.bbox.get_score(),
        }
        summary_res, = session.run([summs['summary_t']], feed_dict=feed_dict)
        our_state['summary_writer'].add_summary(summary_res, step)
        our_state['summary_writer'].flush()

    # cleanup batch
    our_state['batch'] = our_state['batch'][-1:]
    our_state['reward'] = our_state['reward'][-1:]


def dumb_reward_hook(our_state, reward, last_round):
    # prevent state history to eat all memory
    our_state['batch'].pop()


def make_forward_net(state_t):
    """
    Create forward network which maps state into Q-value for all actions
    :param state_t:
    :return:
    """

    with tf.name_scope("L0"):
        w = tf.Variable(tf.random_normal((infra.n_features, L1_SIZE), mean=0.0, stddev=0.1))
        b = tf.Variable(tf.zeros((L1_SIZE,)))
        l0_out = tf.nn.relu(tf.matmul(state_t, w) + b)

    with tf.name_scope("L1"):
        w = tf.Variable(tf.random_normal((L1_SIZE, L2_SIZE), mean=0.0, stddev=0.1))
        b = tf.Variable(tf.zeros((L2_SIZE,)))
        l1_out = tf.nn.relu(tf.matmul(l0_out, w) + b)

    with tf.name_scope("L2"):
        w = tf.Variable(tf.random_normal((L2_SIZE, L3_SIZE), mean=0.0, stddev=0.1))
        b = tf.Variable(tf.zeros((L3_SIZE,)))
        l2_out = tf.nn.relu(tf.matmul(l1_out, w) + b)

    with tf.name_scope("L3"):
        w = tf.Variable(tf.random_normal((L3_SIZE, infra.n_actions), mean=0.0, stddev=0.1))
        b = tf.Variable(tf.zeros((infra.n_actions,)))
        output = tf.matmul(l2_out, w) + b
        output = tf.squeeze(output)

    return output


def make_loss_and_optimiser(state_t, q_vals_t, forward_t):
    with tf.name_scope("Opt"):
        loss_t = tf.nn.l2_loss(forward_t - q_vals_t)
        optimiser = tf.train.AdamOptimizer(learning_rate=LR)
        opt_t = optimiser.minimize(loss_t)

    return loss_t, opt_t


def make_summaries():
    res = {
        'loss': tf.Variable(0.0, trainable=False, name="loss"),
        'score': tf.Variable(0.0, trainable=False, name="score"),
    }

    for name, var in res.iteritems():
        tf.scalar_summary(name, var)

    res['summary_t'] = tf.merge_all_summaries()
    return res


if __name__ == "__main__":
    log = infra.setup_logging()
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
            'summaries': make_summaries(),
        }

        init = tf.initialize_all_variables()
        session.run(init)

        saver = tf.train.Saver()
        global_step = 1

        while True:
            our_state['summary_writer'] = tf.train.SummaryWriter("logs/step=%03d" % global_step)
            our_state['batch'] = []
            our_state['reward'] = []

            infra.prepare_bbox()
            our_state['alpha'] = ALPHA
            our_state['gamma'] = min(GAMMA, GAMMA * float(global_step-1) / 10)

            log.info("%d: Learning round, gamma=%f", global_step, our_state['gamma'])

            # Learning step
            infra.bbox_loop(our_state, action_hook, reward_hook, verbose=False)
            infra.bbox.finish(verbose=0)

            # Test run
            if DO_TEST:
                our_state['batch'] = []
                our_state['reward'] = []
                log.info("%d: Training round done, perform test run", global_step)
                sys.stdout.flush()
                infra.prepare_bbox()
                our_state['alpha'] = 0.0
                infra.bbox_loop(our_state, action_hook, dumb_reward_hook, verbose=100000)
                infra.bbox.finish(verbose=1)

            log.info("%d: save the model", global_step)
            saver.save(session, "models/model-v2", global_step=global_step)
            global_step += 1

            sys.stdout.flush()

