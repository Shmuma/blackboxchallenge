import cPickle as pickle
import array


def pack_item(item):
    bbox_state, action, reward = item
    return (array.array('f', bbox_state.tolist()).tostring(), action, reward)


def save_replay(items, file_name):
    """
    Save replay data in compact form.
    Every item in items is a .
    :param items: list of triples (bbox_state, action, reward)
    :return:
    """
    data = map(pack_item, items)
    with open(file_name, "w+") as fd:
        pickle.dump(data, fd)
