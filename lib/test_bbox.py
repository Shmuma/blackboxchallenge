import infra, replays

import numpy as np


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


def populate_replay_buffer(replay_buffer, session, states_history, states_t, qvals_t, alpha=0.0,
                           verbose=0, max_steps=None):
    """
    Perform test of neural network using bbox interpreter
    """
    infra.prepare_bbox()
    state = {
        'session': session,
        'history': states_history,
        'alpha': alpha,
        'states_t': states_t,
        'qvals_t': qvals_t,
        'state': [],
        'replay': replay_buffer,
    }

    def action_reward_hook(our_state, bbox_state, rewards, next_states):
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

            qvals, = sess.run([qvals_t], feed_dict={states_t: [our_state['state']]})
            action = np.argmax(qvals)

        if len(our_state['state']) == our_state['history']:
            next_states_head = our_state['state'][-(our_state['history']-1):]
            for next_state in next_states:
                our_state['replay'].append(our_state['state'], rewards, next_states_head + [next_state])

        return action

    _max_steps = None if max_steps is None else max_steps + states_history - 1
    infra.bbox_checkpoints_loop(state, action_reward_hook, verbose=verbose, max_steps=_max_steps)
    return infra.bbox.get_score()
