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
    filled = []
    results = [[first_val]]
    for delta, start, stop in stripes[5]:
        f, res = _transform_striped(value, delta=delta, start=start, stop=stop)
        filled.append(f)
        results.append(res)
    if first_bucket or any(filled):
        left = 0.0
    else:
        left = value
    results.append([left])
    return np.concatenate(results)


def _transform_06(value):
    first_bucket = value < -0.655
    first_val = 1.0 if first_bucket else 0.0
    filled = []
    results = [[first_val]]
    for delta, start, stop in stripes[6]:
        f, res = _transform_striped(value, delta=delta, start=start, stop=stop)
        filled.append(f)
        results.append(res)
    if first_bucket or any(filled):
        left = 0.0
    else:
        left = value
    results.append([left])
    return np.concatenate(results)


def _transform_07(value):
    first_bucket = value < -0.6
    filled = []
    results = [[float(first_bucket)]]
    for delta, start, stop in stripes[7]:
        f, res = _transform_striped(value, delta=delta, start=start, stop=stop)
        filled.append(f)
        results.append(res)
    if first_bucket or any(filled):
        left = 0.0
    else:
        left = value
    results.append([left])
    return np.concatenate(results)


def _transform_08(value):
    first_bucket = value < -0.6
    filled = []
    results = [[float(first_bucket)]]
    for delta, start, stop in stripes[8]:
        f, res = _transform_striped(value, delta=delta, start=start, stop=stop)
        filled.append(f)
        results.append(res)
    if first_bucket or any(filled):
        left = 0.0
    else:
        left = value
    results.append([left])
    return np.concatenate(results)



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
    if first > 0.5:
        return -0.6567607522

    for stripe, (delta, start, _) in enumerate(stripes[5]):
        bound = data[1+stripe*60:1+(stripe+1)*60]
        if np.count_nonzero(bound) == 1:
            idx = np.nonzero(bound)[0][0]
            return start + idx * delta
    return data[-1]


def _reverse_06(data):
    assert len(data) == sizes[5]
    assert np.count_nonzero(data) == 1

    first = data[0]
    if first > 0.5:
        return -0.6600022316

    for stripe, (delta, start, _) in enumerate(stripes[6]):
        bound = data[1+stripe*60:1+(stripe+1)*60]
        if np.count_nonzero(bound) == 1:
            idx = np.nonzero(bound)[0][0]
            return start + idx * delta
    return data[-1]


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