import numpy as np
import array

ORIGIN_N_FEATURES = 36
RESULT_N_FEATURES = 35*32 + 1


def transform(state):
    """
    All features except last transfomed into bits value, last left as is
    :param state:
    :return: transformed numpy vector
    """
    arr = array.array('f', state[:-1])
    last = state[-1]

    data = []

    a_ints = array.array('I')
    a_ints.fromstring(arr.tostring())
    for int_val in a_ints.tolist():
        data += map(int, np.binary_repr(int_val, width=32))

    data.append(last)
    return np.array(data, dtype=np.float32)