import tensorflow as tf

import infra

L1_SIZE = 256
L2_SIZE = 256
L3_SIZE = 128


def make_opt(loss_t, learning_rate, decay_every_steps=10000):
    global_step = tf.Variable(0, trainable=False, name="global_step")

    if decay_every_steps is not None:
        exp_learning_rate = tf.train.exponential_decay(
                learning_rate, global_step, decay_every_steps, 0.9, staircase=True)
        tf.scalar_summary("learningRate", exp_learning_rate)
    else:
        exp_learning_rate = learning_rate

    optimiser = tf.train.AdamOptimizer(learning_rate=exp_learning_rate)
    opt_t = optimiser.minimize(loss_t, global_step=global_step)
    return opt_t, optimiser, global_step


def get_vars(trainable, only_weights=False):
    layers = ["L0", "L1", "L2", "L3"]
    l_suffix = "_T" if trainable else "_R"
    names = []
    for l in layers:
        names.append(l + l_suffix + "/w:0")
        if not only_weights:
            names.append(l + l_suffix + "/b:0")
    vars = {}
    for v in tf.all_variables():
        if v.name in names:
            vars[v.name] = v
    return [(name, vars[name]) for name in names]


def make_sync_nets():
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


def make_summaries(loss_t, optimiser):
    res = {
        'loss':             tf.Variable(0.0, trainable=False, name="loss"),
        'speed':            tf.Variable(0.0, trainable=False, name="speed"),
    }

    for name, var in res.iteritems():
        tf.scalar_summary(name, var)

    # weights and gradients summary
    _, v2_vars = zip(*get_vars(trainable=True))
    grads = optimiser.compute_gradients(loss_t, v2_vars)
    for grad, var in grads:
        tf.scalar_summary("magnitude_" + var.name, tf.sqrt(tf.nn.l2_loss(var)))
        tf.scalar_summary("magnitudeGrad_" + var.name, tf.sqrt(tf.nn.l2_loss(grad)))

    res['summary_t'] = tf.merge_all_summaries()
    return res


def make_vars(n_features):
    state_t = tf.placeholder(tf.float32, (None, n_features), name="state")
    rewards_t = tf.placeholder(tf.float32, (None, infra.n_actions), name="reward")
    next_state_t = tf.placeholder(tf.float32, (None, infra.n_actions, n_features), name="next_state")

    return state_t, rewards_t, next_state_t


def leaky_relu(x_t, name, alpha=0.01, summary=True):
    res_t = tf.maximum(x_t * alpha, x_t)
    if summary:
        negatives_t = tf.reduce_mean(tf.to_float(tf.less(res_t, 0.0)))
        name = "%s/negatives" % name
        tf.contrib.layers.summaries._add_scalar_summary(negatives_t, name)
    return res_t


def make_forward_net(states_t, is_main_net, n_features, dropout_keep_prob=0.5):
    if not is_main_net:
        states_t = tf.reshape(states_t, (-1, n_features))

    w_attrs = {
        'trainable': is_main_net,
        'name': 'w',
    }

    b_attrs = {
        'trainable': is_main_net,
        'name': 'b',
    }

    # zero initialisation helps on first stages of learning to reduce randomness in Q values
    if is_main_net:
        init = tf.contrib.layers.xavier_initializer()
        suff = "_T"
        dropout = True
    else:
        init = tf.zeros
        suff = "_R"
        dropout = False

    with tf.name_scope("L0" + suff):
        w = tf.Variable(init((n_features, L1_SIZE)), **w_attrs)
        b = tf.Variable(tf.zeros((L1_SIZE,)), **b_attrs)
        v = tf.matmul(states_t, w) + b
        l0_out = leaky_relu(v, name="L0", summary=is_main_net)

    with tf.name_scope("L1" + suff):
        w = tf.Variable(init((L1_SIZE, L2_SIZE)), **w_attrs)
        b = tf.Variable(tf.zeros((L2_SIZE,)), **b_attrs)
        v = tf.matmul(l0_out, w) + b
        if dropout:
            v = tf.nn.dropout(v, dropout_keep_prob)
        l1_out = tf.nn.relu(v)
        if is_main_net:
            tf.contrib.layers.summarize_activation(l1_out)

    with tf.name_scope("L2" + suff):
        w = tf.Variable(init((L2_SIZE, L3_SIZE)), **w_attrs)
        b = tf.Variable(tf.zeros((L3_SIZE,)), **b_attrs)
        v = tf.matmul(l1_out, w) + b
        if dropout:
            v = tf.nn.dropout(v, dropout_keep_prob)
        l2_out = tf.nn.relu(v)
        if is_main_net:
            tf.contrib.layers.summarize_activation(l2_out)

    with tf.name_scope("L3" + suff):
        w = tf.Variable(init((L3_SIZE, infra.n_actions)), **w_attrs)
        b = tf.Variable(tf.zeros((infra.n_actions,)), **b_attrs)
        output = tf.add(tf.matmul(l2_out, w), b, name="qvals")

    return output


def make_loss(batch_size, gamma, qvals_t, rewards_t, next_qvals_t, n_actions=4, l2_reg=0.0):
    max_qvals = tf.reduce_max(next_qvals_t, 1) * gamma
    q_ref = tf.add(rewards_t, tf.reshape(max_qvals, (batch_size, n_actions)), name="q_ref")
    loss_vec = tf.reduce_sum(tf.abs(qvals_t - q_ref), 1)
    error = tf.nn.l2_loss(qvals_t - q_ref, name="loss_err")

    regularize = tf.contrib.layers.l2_regularizer(l2_reg)
    _, vars = zip(*get_v2_vars(trainable=True, only_weights=True))
    reg_vars = map(regularize, vars)
    l2_error = tf.add_n(reg_vars, name="loss_l2")

    tf.contrib.layers.summarize_tensors([l2_error, error])

    return error + l2_error, loss_vec
