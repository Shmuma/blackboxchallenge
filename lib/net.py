import tensorflow as tf

import infra

L1_SIZE = 512
L2_SIZE = 512
L3_SIZE = 512


def make_vars():
    state_t = tf.placeholder(tf.float32, (None, infra.n_features), name="State")
    q_vals_t = tf.placeholder(tf.float32, (None, infra.n_actions), name="QVals")

    return state_t, q_vals_t


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


def make_loss_and_optimiser(learning_rate, state_t, q_vals_t, forward_t):
    with tf.name_scope("Opt"):
        loss_t = tf.nn.l2_loss(forward_t - q_vals_t)
        optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
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


def make_vars_v2(states_history):
    state_t = tf.placeholder(tf.float32, (None, infra.n_features * states_history), name="state")
    action_t = tf.placeholder(tf.int32, (None, 1), name="action")
    reward_t = tf.placeholder(tf.float32, (None, 1), name="reward")
    next_state_t = tf.placeholder(tf.float32, (None, infra.n_features * states_history), name="next_state")

    return state_t, action_t, reward_t, next_state_t


def make_forward_net_v2(states_history, states_t, is_trainable):
    suff = "_T" if is_trainable else "_R"
    with tf.name_scope("L0" + suff):
        w = tf.Variable(tf.random_normal((infra.n_features * states_history, L1_SIZE),
                                         mean=0.0, stddev=0.1),
                        trainable=is_trainable, name="w")
        b = tf.Variable(tf.zeros((L1_SIZE,)), trainable=is_trainable, name="b")
        l0_out = tf.nn.relu(tf.matmul(states_t, w) + b)

    with tf.name_scope("L1" + suff):
        w = tf.Variable(tf.random_normal((L1_SIZE, L2_SIZE),
                                         mean=0.0, stddev=0.1),
                        trainable=is_trainable, name="w")
        b = tf.Variable(tf.zeros((L2_SIZE,)), trainable=is_trainable, name="b")
        l1_out = tf.nn.relu(tf.matmul(l0_out, w) + b)

    with tf.name_scope("L2" + suff):
        w = tf.Variable(tf.random_normal((L2_SIZE, L3_SIZE), mean=0.0, stddev=0.1),
                        trainable=is_trainable, name="w")
        b = tf.Variable(tf.zeros((L3_SIZE,)), trainable=is_trainable, name="b")
        l2_out = tf.nn.relu(tf.matmul(l1_out, w) + b)

    with tf.name_scope("L3" + suff):
        w = tf.Variable(tf.random_normal((L3_SIZE, infra.n_actions),
                                         mean=0.0, stddev=0.1),
                        trainable=is_trainable, name="w")
        b = tf.Variable(tf.zeros((infra.n_actions,)), trainable=is_trainable, name="b")
        output = tf.matmul(l2_out, w) + b
        output = tf.squeeze(output)

    return output


def make_loss_v2(batch_size, gamma, qvals_t, actions_t, rewards_t, next_qvals_t, n_actions=4):
    # extract qvalues using actions as index.
    # To do this, flatten qvalues into single vector
    q_flat = tf.reshape(qvals_t, (batch_size * n_actions, ))
    act_idx = tf.range(batch_size) * n_actions + actions_t
    # vector with qvalues from taken actions
    q_actions = tf.gather(q_flat, act_idx)
    # reference q_values from Bellman's equation
    q_ref = rewards_t + gamma * tf.reduce_max(next_qvals_t, 1)
    # error
    error = tf.reduce_mean(tf.pow(q_actions - q_ref, 2), name="error")
    return error


def make_opt_v2(loss_t, learning_rate):
    global_step = tf.Variable(0, trainable=False, name="global_step")
    exp_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 1000, 0.9, staircase=True)
    tf.scalar_summary("learningRate", exp_learning_rate)

    optimiser = tf.train.AdamOptimizer(learning_rate=exp_learning_rate)
    opt_t = optimiser.minimize(loss_t)
    return opt_t


def make_sync_nets_v2():
    sync_vars = [
        ("L0_T/w", "L0_R/w"),
        ("L0_T/b", "L0_R/b"),
        ("L1_T/w", "L1_R/w"),
        ("L1_T/b", "L1_R/b"),
        ("L2_T/w", "L2_R/w"),
        ("L2_T/b", "L2_R/b"),
        ("L3_T/w", "L3_R/w"),
        ("L3_T/b", "L3_R/b"),
    ]
    vars = {}

    for v1, v2 in sync_vars:
        vars[v1+":0"] = None
        vars[v2+":0"] = None

    for v in tf.all_variables():
        if v.name in vars:
            vars[v.name] = v

    ops = []
    for src_name, dst_name in sync_vars:
        src = vars[src_name+":0"]
        dst = vars[dst_name+":0"]
        ops.append(dst.assign(src))

    return tf.group(*ops)


def make_summaries_v2():
    res = {
        'loss': tf.Variable(0.0, trainable=False, name="loss"),
        'score': tf.Variable(0.0, trainable=False, name="score"),
        'speed': tf.Variable(0.0, trainable=False, name="speed"),
    }

    for name, var in res.iteritems():
        tf.scalar_summary(name, var)

    res['summary_t'] = tf.merge_all_summaries()
    return res