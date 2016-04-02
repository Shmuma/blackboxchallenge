import infra

import numpy as np


def test_net(session, states_history, states_t, qvals_t, alpha=0.0, verbose=0):
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
        'state': []
    }

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

        return action

    def reward_hook(our_state, reward, last_round):
        pass

    infra.bbox_loop(state, action_hook, reward_hook, verbose=verbose)
    return infra.bbox.get_score()
