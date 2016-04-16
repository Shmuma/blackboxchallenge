import numpy as np
import math

ORIGIN_N_FEATURES = 36

def transform(state):
    """
    Perform transformation of state vector to our representation
    :param state: numpy array of 36 values from bbox state
    :return: another numpy array with transformed state
    """
    result = []
    for feat in xrange(ORIGIN_N_FEATURES):
        if feat in transforms:
            result.append(transforms[feat](state[feat]))
        else:
            result.append([state[feat]])
    return np.concatenate(result)


def transformed_size():
    """
    Return final length of vector
    :return:
    """
    return 36 - len(sizes) + sum(sizes.values())


def _transform_00(value):
    filled_1, res_1 = _transform_striped(value, delta=0.0029091158169317999, start=-0.7769826651, stop=-0.6053448319)
    filled_2, res_2 = _transform_striped(value, delta=0.0046108435779661019, start= 1.2650883198, stop= 1.5371280909)
    filled_3, res_3 = _transform_striped(value, delta=0.0058182296101694873, start= 2.7139606476, stop= 3.0572361946)
    if filled_1 or filled_2 or filled_3:
        first = 0.0
    else:
        first = value
    return np.concatenate([[first], res_1, res_2, res_3])


def _transform_05(value):
    first_bucket = value < -0.655
    first_val = 1.0 if first_bucket else 0.0
    filled_1, res_1 = _transform_striped(value, delta=0.00097969932033898461, start=0.5622291565, stop=0.6200314164)
    filled_2, res_2 = _transform_striped(value, delta=0.00155278787796609800, start=1.2752926350, stop=1.3669071198)
    if first_bucket or filled_1 or filled_2:
        left = 0.0
    else:
        left = value
    return np.concatenate([[first_val], res_1, res_2, [left]])


def _transform_35(value):
    # resulting value is in range [-11..11]
    int_val = int(value * 10.0)
    assert -12 < int_val < 12

    # perform one-hot encoding
    result = np.zeros((23,))
    result[int_val + 11] = 1
    return result


def _split_bound_func(bound):
    def fun(value):
        res = np.zeros((2,))
        if value < bound:
            res[0] = value
        else:
            res[1] = value
        return res
    return fun


# dictionary with resulting feature sizes
sizes = {
    0:  1 + 60 + 60 + 60,
    1:  2,
    2:  2,
    3:  2,
    4:  2,
    5: 1 + 60 + 60 + 1,
    35: 23
}

transforms = {
    0: _transform_00,
    1: _split_bound_func(0.0),
    2: _split_bound_func(0.0),
    3: _split_bound_func(0.0),
    4: _split_bound_func(0.0),
    5: _transform_05,
    35: _transform_35
}


def _transform_striped(value, delta, start, stop):
    """
    Perform striped decoding of value
    :param value: value to decode
    :param delta: delta step for stripe
    :param start: first stripe
    :param stop: last stripe
    :return: tuple from (filled_bool, one-hot array)
    """
    count = int(round((stop-start) / delta + 1))
    res = np.zeros((count,))
    if value < start - delta/2:
        return False, res
    ofs = 0
    bound = start + delta/2
    filled = False
    while bound < stop + delta:
        if value < bound:
            res[ofs] = 1.0
            filled = True
            break
        ofs += 1
        bound += delta
    return filled, res


# below code exists only for debugging and testing purposes -- reverse transformation of features back to values
def _reverse_00(data):
    assert len(data) == sizes[0]
    assert np.count_nonzero(data) == 1
    first_val = data[0]
    bound_1 = data[1:1+60]
    bound_2 = data[1+60:1+60+60]
    bound_3 = data[1+60+60:1+60+60+60]
    if abs(first_val) > 1e-5:
        return first_val
    if np.count_nonzero(bound_1) == 1:
        idx = np.nonzero(bound_1)[0][0]
        return -0.7769826651 + idx * 0.0029091158169317999
    if np.count_nonzero(bound_2) == 1:
        idx = np.nonzero(bound_2)[0][0]
        return 1.2650883198 + idx * 0.0046108435779661019
    if np.count_nonzero(bound_3) == 1:
        idx = np.nonzero(bound_3)[0][0]
        return 2.7139606476 + idx * 0.0058182296101694873
    assert False


def _unsplit_bound(data):
    """
    Perform reverse of _split_bound_func(bound) application - join two values together.
    To reverse split, just sum data array
    :param data:
    :return:
    """
    assert len(data) == 2
    assert np.count_nonzero(data) == 1
    return data.sum()


reverse_transforms = {
    0: _reverse_00,
    1: _unsplit_bound,
    2: _unsplit_bound,
    3: _unsplit_bound,
    4: _unsplit_bound
}