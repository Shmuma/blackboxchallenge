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
    35: 23
}

transforms = {
    35: _transform_35
}

