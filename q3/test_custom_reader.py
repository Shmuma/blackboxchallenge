import sys
sys.path.append("..")

from lib import replays, infra

import logging as log
import tensorflow as tf


def make_input_pipeline():
    queue = tf.train.range_input_producer(10, shuffle=False)
    value = queue.dequeue()
    print value
    return value


def action_hook(our_state, bbox_state):
    action = 0
    our_state['state'].append(bbox_state)
    our_state['state'] = our_state['state'][-our_state['history']:]
    our_state['action'] = action
    return action


def reward_hook(our_state, reward, last_round):
    if not last_round and len(our_state['state']) == our_state['history']:
        next_state = our_state['state'][-(our_state['history']-1):]
        next_state.append(infra.bbox.get_state())
        our_state['replay'].append(our_state['state'], our_state['action'], reward, next_state)


if __name__ == "__main__":
    input_t = make_input_pipeline()
    replay_buffer = replays.ReplayBuffer(1000, 50)

    infra.setup_logging()
    infra.prepare_bbox()

    state = {
        'replay': replay_buffer,
        'history': 4,
        'state': []
    }

    log.info("Populating replay buffer")

    infra.bbox_loop(state, action_hook, reward_hook, verbose=10, max_steps=200)
    replay_buffer.reshuffle()
    log.info("Replay={replay}".format(replay=replay_buffer))

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

        try:
            for _ in xrange(10):
                batch = replay_buffer.next_batch()
        finally:
            coordinator.request_stop()
            coordinator.join(threads)
