import numpy as np

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
    return _transform_striped(value, delta=0.0029091158169317999, start=-0.7769826650988442, stop=-0.605344831899868)


def _transform_35(value):
    # resulting value is in range [-11..11]
    int_val = int(value * 10.0)
    assert -12 < int_val < 12

    # perform one-hot encoding
    result = np.zeros((23,))
    result[int_val + 11] = 1
    return result


# dictionary with resulting feature sizes
sizes = {
    0:  61,
    35: 23
}

transforms = {
    0: _transform_00,
    35: _transform_35
}


def _transform_striped(value, delta, start, stop, fill_value = 0.0):
    """
    Perform striped decoding of value
    :param value: value to decode
    :param delta: delta step for stripe
    :param start: first stripe
    :param stop: last stripe
    :return: numpy array with first 'value' or fill_value and rest are one-hot encoding of range
    """
    count = int((stop-start) / delta + 1)
    res = np.zeros((count + 1,))
    ofs = 1
    bound = start + delta/2
    filled = False
    while bound < stop + delta:
        if value < bound:
            res[ofs] = 1.0
            filled = True
            break
        ofs += 1
        bound += delta

    if not filled:
        res[0] = value
    else:
        res[0] = fill_value

    return res