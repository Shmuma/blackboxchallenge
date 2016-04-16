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
    filled_3, res_3 = _transform_striped(value, delta=0.00195939864067796920, start=1.7812191248, stop=1.8968236446)
    filled_4, res_4 = _transform_striped(value, delta=0.00227479207288135530, start=2.1736462116, stop=2.3078589439)
    filled_5, res_5 = _transform_striped(value, delta=0.00253248618813559780, start=2.4942824841, stop=2.6436991692)
    filled_6, res_6 = _transform_striped(value, delta=0.00275036440169491680, start=2.7653765678, stop=2.9276480675)
    filled_7, res_7 = _transform_striped(value, delta=0.00293910301355931970, start=3.0002088547, stop=3.1736159325)
    if first_bucket or filled_1 or filled_2 or filled_3 or filled_4 or filled_5 or filled_6 or filled_7:
        left = 0.0
    else:
        left = value
    return np.concatenate([[first_val], res_1, res_2, res_3, res_4, res_5, res_6, res_7, [left]])


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
    5: 1 + 60*7 + 1,
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


def _reverse_05(data):
    assert len(data) == sizes[5]
    assert np.count_nonzero(data) == 1

    first = data[0]
    bound_1 = data[1:1+60]
    bound_2 = data[1+60:1+60+60]
    bound_3 = data[1+60*2:1+60*3]
    bound_4 = data[1+60*3:1+60*4]
    bound_5 = data[1+60*4:1+60*5]
    bound_6 = data[1+60*5:1+60*6]
    bound_7 = data[1+60*6:1+60*7]
    left = data[1+60*7]

    if first > 0.5:
        return -0.6567607522
    if np.count_nonzero(bound_1) == 1:
        idx = np.nonzero(bound_1)[0][0]
        return 0.5622291565 + idx * 0.00097969932033898461
    if np.count_nonzero(bound_2) == 1:
        idx = np.nonzero(bound_2)[0][0]
        return 1.2752926350 + idx * 0.00155278787796609800
    if np.count_nonzero(bound_3) == 1:
        idx = np.nonzero(bound_3)[0][0]
        return 1.7812191248 + idx * 0.00195939864067796920
    if np.count_nonzero(bound_4) == 1:
        idx = np.nonzero(bound_4)[0][0]
        return 2.1736462116 + idx * 0.00227479207288135530
    if np.count_nonzero(bound_5) == 1:
        idx = np.nonzero(bound_5)[0][0]
        return 2.4942824841 + idx * 0.00253248618813559780
    if np.count_nonzero(bound_6) == 1:
        idx = np.nonzero(bound_6)[0][0]
        return 2.7653765678 + idx * 0.00275036440169491680
    if np.count_nonzero(bound_7) == 1:
        idx = np.nonzero(bound_7)[0][0]
        return 3.0002088547 + idx * 0.00293910301355931970
    return left


def _reverse_35(data):
    assert len(data) == 23
    assert np.count_nonzero(data) == 1
    idx = np.nonzero(data)[0][0]
    return float(idx-11) / 10.0


reverse_transforms = {
    0: _reverse_00,
    1: _unsplit_bound,
    2: _unsplit_bound,
    3: _unsplit_bound,
    4: _unsplit_bound,
    5: _reverse_05,
    35: _reverse_35,
}