import numpy as np
from scipy.sparse import csr_matrix
import math


# dictionary with resulting feature sizes
sizes = {
    0:  60*3 + 1,
    1:  2,
    2:  2,
    3:  2,
    4:  2,
    5:  1 + 60*4 + 1,
    6:  1 + 60*4 + 1,
    7:  1 + 60*4 + 1,
    8:  1 + 60*4 + 1,
    9:  1 + 60*4 + 1,
    10: 1 + 60*4 + 1,
    35: 23
}


def transformed_size():
    """
    Return final length of vector
    :return:
    """
    return 36 - len(sizes) + sum(sizes.values())


ORIGIN_N_FEATURES = 36
RESULT_N_FEATURES = transformed_size()


def transform(state):
    """
    Perform transformation of state vector to our representation
    :param state: numpy array of 36 values from bbox state
    :return: dict with features
    """
    res = []
    ofs = 0
    for feat in xrange(ORIGIN_N_FEATURES):
        if feat in transforms:
            idx, val = transforms[feat](state[feat])
            res.append((idx + ofs, val))
            ofs += sizes[feat]
        else:
            res.append((ofs, float(state[feat])))
            ofs += 1

    idx, vals = zip(*res)
    return np.array(idx, dtype=np.int16), np.array(vals, dtype=np.float32)


def _transform_00(value):
    return _transform_bound_and_stripes(value, stripes=stripes[0])

def _transform_05(value):
    return _transform_bound_and_stripes(value, stripes=stripes[5])

def _transform_06(value):
    return _transform_bound_and_stripes(value, stripes=stripes[6])

def _transform_07(value):
    return _transform_bound_and_stripes(value, stripes=stripes[7])

def _transform_08(value):
    return _transform_bound_and_stripes(value, stripes=stripes[8])

def _transform_09(value):
    return _transform_bound_and_stripes(value, stripes=stripes[9])

def _transform_10(value):
    return _transform_bound_and_stripes(value, stripes=stripes[10])


def _transform_bound_and_stripes(value, stripes, eps=1e-6):
    output_index = 0
    for delta, start, stop in stripes:
        # if delta is none, encode value as single stripe
        if delta is None:
            if start - eps <= value <= start + eps:
                filled_index = 0
            else:
                filled_index = None
            total_size = 1
        else:
            filled_index, total_size = _transform_striped(value, delta=delta, start=start, stop=stop)
        if filled_index is not None:
            return (filled_index + output_index, 1.0)
        output_index += total_size
    return output_index, value


def _reverse_bound_and_stripes(data, stripes):
    idx, val = data
    ofs = 0
    stripe = 0

    for delta, start, _ in stripes:
        if delta is None:
            if ofs+stripe*60 == idx:
                return start
            ofs += 1
        else:
            if ofs+stripe*60 <= idx < ofs+(stripe+1)*60:
                return start + delta * (idx - ofs+stripe*60)
            stripe += 1
    return val


def _transform_35(value):
    # resulting value is in range [-11..11]
    int_val = int(value * 10.0)
    assert -12 < int_val < 12
    return (int_val + 11, 1.0)


def _split_bound_func(bound):
    def fun(value):
        if value < bound:
            return (0, float(value))
        else:
            return (1, float(value))
    return fun


# all of them should return tuple with (idx, value) of sparse representation
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
    9: _transform_09,
    10: _transform_10,
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
        (None,                  -0.6567607522, None),
        (0.0009796993203389846,  0.5622291565, 0.6200314164),
        (0.0015527878779660980,  1.2752926350, 1.3669071198),
        (0.0019593986406779692,  1.7812191248, 1.8968236446),
        (0.0022747920728813553,  2.1736462116, 2.3078589439),
#        (0.0025324861881355978, 2.4942824841, 2.6436991692),
#        (0.0027503644016949168, 2.7653765678, 2.9276480675),
#        (0.0029391030135593197, 3.0002088547, 3.1736159325),
    ],

    6: [
        (None,                  -0.6600022316, None),
        (0.0009572505949152544,  0.5395243168, 0.5960021019),
        (0.0015172049152542358,  1.2412025928, 1.3307176828),
        (0.0019144991694915242,  1.7390509844, 1.8520064354),
        (0.0022226673067796670,  2.1252121925, 2.2563495636),
#        (0.0024744534898305092, 2.4407291412, 2.5867218971),
#        (0.0026873410762711858, 2.7074947357, 2.8660478592),
#        (0.0028717437033898212, 2.9385778904, 3.1080107689),
    ],

    7: [
        (None,                  -0.6103122234,  None),
        (0.0003188861118638514, -0.2078952789, -0.1890810000),
        (0.0005054219745762448,  0.0275035407,  0.0573234372),
        (0.0006377722237287495,  0.1945216805,  0.2321502417),
        (0.0007404306169492319,  0.3240709901,  0.3677563965),
    ],

    8: [
        (None,                   -0.6154794097,  None),
        (0.00031347825423676517, -0.2171127051, -0.1986174881),
        (0.00049685071186441980,  0.0159168709,  0.0452310629),
        (0.00062695650677974492,  0.1812539995,  0.2182444334),
        (0.00072787373728827034,  0.3094994128,  0.3524439633),
    ],

    9: [
        (0.0015689952898412618, -0.806575179100036, -0.7140044569994015),
        (None,                   1.293757915496888,  None),
        (6.747447965817151e-05,  1.380100250200044,  1.3840812444998756),
        (0.00010694285593263897, 1.4306073189000015, 1.4369169474000272),
        (0.00013494693559368089, 1.4664427041999817, 1.4744045734000089),
    ],

    10: [
        (0.0015545467203457062, -0.8129093647002773, -0.7211911081998806),
        (None,                   1.2818331718288425,  None),
        (6.6852166106599764e-05, 1.367971777899722,   1.3719160557000114),
        (0.00010595887288113288, 1.4183596372999998,  1.4246112107999866),
        (0.00013370433050833354, 1.4541103840000127,  1.4619989395000044),
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
    if value < start - delta/2:
        return None, count
    if value > stop + delta/2:
        return None, count
    ofs = 0
    bound = start + delta/2
    while bound < stop + delta:
        if value < bound:
            return ofs, count
        ofs += 1
        bound += delta
    return None, count


# below code exists only for debugging and testing purposes -- reverse transformation of features back to values
def _reverse_00(data):
    return _reverse_bound_and_stripes(data, stripes=stripes[0])


def _unsplit_bound(data):
    """
    Perform reverse of _split_bound_func(bound) application - join two values together.
    :param data:
    :return:
    """
    assert len(data) == 2
    return data[1]


def _reverse_05(data):
    return _reverse_bound_and_stripes(data, stripes=stripes[5])

def _reverse_06(data):
    return _reverse_bound_and_stripes(data, stripes=stripes[6])

def _reverse_07(data):
    return _reverse_bound_and_stripes(data, stripes=stripes[7])

def _reverse_08(data):
    return _reverse_bound_and_stripes(data, stripes=stripes[8])

def _reverse_09(data):
    return _reverse_bound_and_stripes(data, stripes=stripes[9])

def _reverse_10(data):
    return _reverse_bound_and_stripes(data, stripes=stripes[10])


def _reverse_35(data):
    idx, _ = data
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
    9: _reverse_09,
    10: _reverse_10,
    35: _reverse_35,
}


def to_dense(sparse):
    res = np.zeros((RESULT_N_FEATURES,))
    apply_dense(res, sparse)
    return res

def apply_dense(vector, sparse):
    vector[sparse[0]] = sparse[1]
    return vector
