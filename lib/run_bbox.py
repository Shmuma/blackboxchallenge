import infra, replays, features

import numpy as np

from lib import net_light


def test_net(session, states_history, states_t, qvals_t, alpha=0.0, verbose=0, save_prefix=None):
    """
    Perform test of neural network using bbox interpreter
    :param session:
    :param states_history:
    :param states_t:
    :param qvals_t:
    :return:
    """
    infra.prepare_bbox()
    state = {
        'session': session,
        'history': states_history,
        'alpha': alpha,
        'states_t': states_t,
        'qvals_t': qvals_t,
        'state': [],
        'writer': None,
    }

    if save_prefix is not None:
        state['writer'] = replays.ReplayWriter(save_prefix)

    def action_hook(our_state, bbox_state):
        # save state, keep only fixed amount of states
        our_state['state'].append(bbox_state)
        our_state['state'] = our_state['state'][-our_state['history']:]

        # make decision about action
        if np.random.random() < our_state['alpha'] or len(our_state['state']) < our_state['history']:
            action = np.random.randint(0, infra.n_actions, 1)[0]
        else:
            sess = our_state['session']
            qvals_t = our_state['qvals_t']
            states_t = our_state['states_t']

            state_v = np.array(our_state['state'])
            state_v = state_v.reshape((1, state_v.shape[0] * state_v.shape[1]))

            qvals, = sess.run([qvals_t], feed_dict={states_t: state_v})
            action = np.argmax(qvals)

        our_state['action'] = action

        return action

    def reward_hook(our_state, reward, last_round):
        if last_round or our_state['writer'] is None:
            return

        if len(our_state['state']) == our_state['history']:
            next_state = our_state['state'][-(our_state['history']-1):]
            next_state.append(infra.bbox.get_state())
            our_state['writer'].append_small(our_state['state'], our_state['action'],
                                             reward, next_state)

    infra.bbox_loop(state, action_hook, reward_hook, verbose=verbose)
    return infra.bbox.get_score()


def populate_replay_buffer(replay_buffer, session, states_t, qvals_t, alpha=0.0,
                           verbose=0, max_steps=None):
    """
    Refill replay buffer
    """
    infra.prepare_bbox()
    state = {
        'session': session,
        'alpha': alpha,
        'states_t': states_t,
        'qvals_t': qvals_t,
        'state': [],
        'replay': replay_buffer,
        'appended': 0
    }

    def action_reward_hook(our_state, bbox_state, rewards, next_states):
        # save state, keep only fixed amount of states
        our_state['state'] = bbox_state

        # make decision about action
        if np.random.random() < our_state['alpha']:
            action = np.random.randint(0, infra.n_actions, 1)[0]
        else:
            sess = our_state['session']
            qvals_t = our_state['qvals_t']
            states_t = our_state['states_t']

            state = features.transform(our_state['state'], infra.bbox.get_time())
            qvals, = sess.run([qvals_t], feed_dict={states_t: [state]})
            action = np.argmax(qvals)

        our_state['replay'].append(our_state['state'], rewards, next_states)
        our_state['appended'] += 1

        return action

    if max_steps is None:
        infra.bbox_checkpoints_loop(state, action_reward_hook, verbose=verbose, max_steps=max_steps)
        score = infra.bbox.get_score()
        avg_score = score / infra.bbox.get_time()
    else:
        score = 0
        steps = 0
        while state['appended'] < max_steps:
            infra.bbox_checkpoints_loop(state, action_reward_hook, verbose=verbose, max_steps=max_steps)
            score += infra.bbox.get_score()
            steps += infra.bbox.get_time()
            infra.bbox.reset_level()
        avg_score = score / steps
    return score, avg_score


def test_performance(session, states_t, qvals_t, alpha=0.0, verbose=0, max_steps=None, test_level=False,
                     feats_tr_post=None, step_hook=None, cache_steps=None):
    """
    Perform test of neural network using bbox interpreter

    args:
    - feats_tr_post: features transformation applied after main transformation
    - step_hook: optional function without arguments to be called every step
    - cache_steps: cache steps for a given amount of steps
    """
    infra.prepare_bbox(test_level=test_level)
    state = {
        'session': session,
        'alpha': alpha,
        'states_t': states_t,
        'qvals_t': qvals_t,
        'cached_action': None,
        'cached_counter': 0,
        'reward_history': [],
    }

    def action_hook(our_state, bbox_state):
        if step_hook is not None:
            step_hook()

        if cache_steps is not None:
            our_state['cached_counter'] -= 1

        # make decision about action
        if np.random.random() < our_state['alpha']:
            action = np.random.randint(0, infra.n_actions, 1)[0]
        else:
            if cache_steps is not None:
                if our_state['cached_counter'] > 0:
                    # print "%d: cached %d (counter %d), rewards %s" % (
                    #     infra.bbox.get_time(), our_state['cached_action'], our_state['cached_counter'],
                    #     our_state['reward_history']
                    # )
                    return our_state['cached_action']
                our_state['cached_counter'] = cache_steps

            sess = our_state['session']
            qvals_t = our_state['qvals_t']
            states_t = our_state['states_t']

            # do a features transformation
            state = features.transform(bbox_state, our_state['reward_history'])
            if feats_tr_post is not None:
                state = feats_tr_post(state)
            qvals, = sess.run([qvals_t], feed_dict={states_t: [state]})
            action = np.argmax(qvals)

            if cache_steps is not None:
                our_state['cached_action'] = action
            # print "%d: action %d calculated, rewards %s" % (infra.bbox.get_time(), action, our_state['reward_history'])

        return action

    def reward_hook(our_state, reward, last_round):
        our_state['reward_history'] = features.push_reward(our_state['reward_history'], reward)

    infra.bbox_loop(state, action_hook, reward_hook, verbose=verbose, max_steps=max_steps)
    score = infra.bbox.get_score()
    avg_score = score / infra.bbox.get_time()
    return score, avg_score


def test_performance_no_tf(network, alpha=0.0, verbose=0, max_steps=None, test_level=False,
                     feats_tr_post=None, step_hook=None):
    """
    Perform test of neural network using bbox interpreter

    args:
    - feats_tr_post: features transformation applied after main transformation
    - step_hook: optional function without arguments to be called every step
    """
    infra.prepare_bbox(test_level=test_level)

    def action_hook(our_state, bbox_state):
        if step_hook is not None:
            step_hook()

        # make decision about action
        if np.random.random() < alpha:
            action = np.random.randint(0, infra.n_actions, 1)[0]
        else:
            # do a features transformation
            state = features.transform(bbox_state, infra.bbox.get_time())
            if feats_tr_post is not None:
                state = feats_tr_post(state)

            qvals = net_light.calc_qvals(network, state)
            action = np.argmax(qvals)

        return action

    def reward_hook(our_state, reward, last_round):
        pass

    infra.bbox_loop(None, action_hook, reward_hook, verbose=verbose, max_steps=max_steps)
    score = infra.bbox.get_score()
    avg_score = score / infra.bbox.get_time()
    return score, avg_score

