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
    return _transform_bound_and_stripes(value, bound=None, stripes=stripes[0])


def _transform_05(value):
    return _transform_bound_and_stripes(value, bound=-0.655, stripes=stripes[5])


def _transform_06(value):
    return _transform_bound_and_stripes(value, bound=-0.655, stripes=stripes[6])


def _transform_07(value):
    return _transform_bound_and_stripes(value, bound=-0.6, stripes=stripes[7])


def _transform_08(value):
    return _transform_bound_and_stripes(value, bound=-0.6, stripes=stripes[8])


def _transform_bound_and_stripes(value, bound, stripes):
    if bound is None:
        first_bucket = False
        results = []
    else:
        first_bucket = value < bound
        results = [[float(first_bucket)]]
    filled = []
    for delta, start, stop in stripes:
        f, res = _transform_striped(value, delta=delta, start=start, stop=stop)
        filled.append(f)
        results.append(res)
    if first_bucket or any(filled):
        left = 0.0
    else:
        left = value
    results.append([left])
    return np.concatenate(results)


def _reverse_bound_and_stripes(data, val_bound, stripes):
    assert np.count_nonzero(data) == 1

    if val_bound is None:
        ofs = 0
    else:
        ofs = 1
        if data[0] > 0.5:
            return val_bound

    for stripe, (delta, start, _) in enumerate(stripes):
        bound = data[ofs+stripe*60:ofs+(stripe+1)*60]
        if np.count_nonzero(bound) == 1:
            idx = np.nonzero(bound)[0][0]
            return start + idx * delta
    return data[-1]


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
    5:  1 + 60 * 4 + 1,
    6:  1 + 60 * 4 + 1,
    7:  1 + 60 * 4 + 1,
    8:  1 + 60 * 4 + 1,
    35: 23
}

transforms = {
    0: _transform_00,
    1: _split_bound_func(0.0),
    2: _split_bound_func(0.0),
    3: _split_bound_func(0.0),
    4: _split_bound_func(0.0),
    5: _transform_05,
    6: _transform_06,
    7: _transform_07,
    8: _transform_08,
    35: _transform_35
}

# stripes are encoded as (delta, start, stop)
stripes = {
    0: [
        (0.0029091158169317999, -0.7769826651, -0.6053448319),
        (0.0046108435779661019,  1.2650883198,  1.5371280909),
        (0.0058182296101694873,  2.7139606476,  3.0572361946),
    ],

    5: [
        (0.0009796993203389846, 0.5622291565, 0.6200314164),
        (0.0015527878779660980, 1.2752926350, 1.3669071198),
        (0.0019593986406779692, 1.7812191248, 1.8968236446),
        (0.0022747920728813553, 2.1736462116, 2.3078589439),
#        (0.0025324861881355978, 2.4942824841, 2.6436991692),
#        (0.0027503644016949168, 2.7653765678, 2.9276480675),
#        (0.0029391030135593197, 3.0002088547, 3.1736159325),
    ],

    6: [
        (0.0009572505949152544, 0.5395243168, 0.5960021019),
        (0.0015172049152542358, 1.2412025928, 1.3307176828),
        (0.0019144991694915242, 1.7390509844, 1.8520064354),
        (0.0022226673067796670, 2.1252121925, 2.2563495636),
#        (0.0024744534898305092, 2.4407291412, 2.5867218971),
#        (0.0026873410762711858, 2.7074947357, 2.8660478592),
#        (0.0028717437033898212, 2.9385778904, 3.1080107689),
    ],

    7: [
        (0.0003188861118638514, -0.2078952789, -0.1890810000),
        (0.0005054219745762448,  0.0275035407,  0.0573234372),
        (0.0006377722237287495,  0.1945216805,  0.2321502417),
        (0.0007404306169492319,  0.3240709901,  0.3677563965),
    ],

    8: [
        (0.00031347825423676517, -0.2171127051, -0.1986174881),
        (0.00049685071186441980,  0.0159168709,  0.0452310629),
        (0.00062695650677974492,  0.1812539995,  0.2182444334),
        (0.00072787373728827034,  0.3094994128,  0.3524439633),
    ]
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
    return _reverse_bound_and_stripes(data, val_bound=None, stripes=stripes[0])


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
    return _reverse_bound_and_stripes(data, val_bound=-0.6567607522, stripes=stripes[5])

def _reverse_06(data):
    return _reverse_bound_and_stripes(data, val_bound=-0.6600022316, stripes=stripes[6])

def _reverse_07(data):
    return _reverse_bound_and_stripes(data, val_bound=-0.6103122234, stripes=stripes[7])

def _reverse_08(data):
    return _reverse_bound_and_stripes(data, val_bound=-0.6154794097, stripes=stripes[8])


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
    6: _reverse_06,
    7: _reverse_07,
    8: _reverse_08,
    35: _reverse_35,
}