import numpy as np


def _stripe_size(feature):
    res = 1
    for stripe in stripes[feature]:
        res += _stripe_lines(stripe)
    return res


def _stripe_lines(stripe):
    delta, start, stop = stripe
    if delta is None:
        return 1
    else:
        return int(round((stop-start) / delta + 1))


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
#        (0.0022747920728813553,  2.1736462116, 2.3078589439),

#        (0.0025324861881355978, 2.4942824841, 2.6436991692),
#        (0.0027503644016949168, 2.7653765678, 2.9276480675),
#        (0.0029391030135593197, 3.0002088547, 3.1736159325),
    ],

    6: [
        (None,                  -0.6600022316, None),
        (0.0009572505949152544,  0.5395243168, 0.5960021019),
        (0.0015172049152542358,  1.2412025928, 1.3307176828),
        (0.0019144991694915242,  1.7390509844, 1.8520064354),
#        (0.0022226673067796670,  2.1252121925, 2.2563495636),

#        (0.0024744534898305092, 2.4407291412, 2.5867218971),
#        (0.0026873410762711858, 2.7074947357, 2.8660478592),
#        (0.0028717437033898212, 2.9385778904, 3.1080107689),
#        (0.0030343850,          3.1424074173, 3.3214361350),
    ],

    7: [
        (None,                  -0.6103122234,  None),
        (0.0003188861118638514, -0.2078952789, -0.1890810000),
        (0.0005054219745762448,  0.0275035407,  0.0573234372),
        (0.0006377722237287495,  0.1945216805,  0.2321502417),
#        (0.0007404306169492319,  0.3240709901,  0.3677563965),

#        (0.0008243075,           0.4299205244,  0.4785546660),
#        (0.0008952274,           0.5194149613,  0.5722333789),
#        (0.0009566570,           0.5969386697,  0.6533814316),
#        (0.0010108449,           0.6653193235,  0.7249591728),
    ],

    8: [
        (None,                   -0.6154794097,  None),
        (0.00031347825423676517, -0.2171127051, -0.1986174881),
        (0.00049685071186441980,  0.0159168709,  0.0452310629),
        (0.00062695650677974492,  0.1812539995,  0.2182444334),
#        (0.00072787373728827034,  0.3094994128,  0.3524439633),
#        (0.0008103287,            0.4142836034,  0.4620929956),
#        (0.0008800444,            0.5028772950,  0.5547999144),
#        (0.0009404350,            0.5796206594,  0.6351063251),
#        (0.0009937010,            0.6473131776,  0.7059415345),
    ],

    9: [
        (0.0015689952898412618, -0.806575179100036, -0.7140044569994015),
        (None,                   1.293757915496888,  None),
        (6.747447965817151e-05,  1.380100250200044,  1.3840812444998756),
        (0.00010694285593263897, 1.4306073189000015, 1.4369169474000272),
#        (0.00013494693559368089, 1.4664427041999817, 1.4744045734000089),
    ],

    10: [
        (0.0015545467203457062, -0.8129093647002773, -0.7211911081998806),
        (None,                   1.2818331718288425,  None),
        (6.6852166106599764e-05, 1.367971777899722,   1.3719160557000114),
        (0.00010595887288113288, 1.4183596372999998,  1.4246112107999866),
#        (0.00013370433050833354, 1.4541103840000127,  1.4619989395000044),
    ],

    11: [
        (0.0003760427,          -2.6104412079, -2.5882546902),
        (0.0005960161,          -2.3250629902, -2.2898980401),
        (0.0007520851,          -2.1225841045, -2.0782110817),
        (0.0008731434,          -1.9655290842, -1.9140136242),
        (0.0009720548,          -1.8372058868, -1.7798546553),
        (0.0010556851,          -1.7287100554, -1.6664246321),
#        (0.0011281292,          -1.6347268820, -1.5681672569),
#        (0.0011920235,          -1.5518276691, -1.4814982804),
    ],

    12: [
        (0.0003740949,          -2.6113741398, -2.5893025398),
        (0.0005929228,          -2.3255391121, -2.2905566692),
        (0.0007481910,          -2.1227362156, -2.0785929487),
        (0.0008686155,          -1.9654299021, -1.9141815901),
        (0.0009670197,          -1.8369013071, -1.7798471451),
        (0.0010502116,          -1.7282317877, -1.6662693024),
#        (0.0011222790,          -1.6340980530, -1.5678835927),
#        (0.0011858451,          -1.5510662794, -1.4811014214),
    ],

    23: [
        (None,                  -2.2445206642,  None),
        (0.0001423601,          -2.0447108746, -2.0363116264),
        (0.0002256389,          -1.9278297424, -1.9145170450),
        (0.0002847223,          -1.8449012041, -1.8281025887),
#        (0.0003305552,          -1.7805768251, -1.7610740662),
#        (0.0003679991,          -1.7280199528, -1.7063080072),
#        (0.0003996582,          -1.6835837364, -1.6600039005),
#        (0.0004270845,          -1.6450914145, -1.6198934317),
#        (0.0004512720,          -1.6111387014, -1.5845136538),
    ],

    24: [
        (None,                  -2.2460646629, None),
        (0.0001417378,          -2.0458757877, -2.0375132561),
        (0.0002246469,          -1.9287724495, -1.9155182838),
        (0.0002834756,          -1.8456865549, -1.8289614916),
#        (0.0003291065,          -1.7812401056, -1.7618228197),
#        (0.0003663867,          -1.7285834551, -1.7069666386),
#        (0.0003979085,          -1.6840629578, -1.6605863571),
#        (0.0004252263,          -1.6454975605, -1.6204092098),
#        (0.0004492978,          -1.6114803553, -1.5849717855),
    ]
}


# dictionary with resulting feature sizes
sizes = {
    0:  _stripe_size(0),
    1:  2,
    2:  2,
    3:  2,
    4:  2,
    5:  _stripe_size(5),
    6:  _stripe_size(6),
    7:  _stripe_size(7),
    8:  _stripe_size(8),
    9:  _stripe_size(9),
    10: _stripe_size(10),
    11: _stripe_size(11),
    12: _stripe_size(12),
    23: _stripe_size(23),
    24: _stripe_size(24),
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
    :return: pair with indices and values (sparse representation)
    """
    res_idx = []
    res_val = []
    ofs = 0
    for feat, value in enumerate(state):
        if feat in transforms:
            idx, val = transforms[feat](value)
            res_idx.append(idx + ofs)
            res_val.append(val)
            ofs += sizes[feat]
        else:
            res_idx.append(ofs)
            res_val.append(value)
            ofs += 1
    return np.array(res_idx, dtype=np.int16), np.array(res_val, dtype=np.float32)


def _transform_stripes_func(feature):
    def _transform(data):
        return _transform_bound_and_stripes(data, stripes=stripes[feature])
    return _transform


def _transform_bound_and_stripes(value, stripes, eps=1e-6):
    output_index = 0
    for stripe in stripes:
        delta, start, stop = stripe

        # if delta is none, encode value as single stripe
        if delta is None:
            if start - eps <= value <= start + eps:
                filled_index = 0
            else:
                filled_index = None
        else:
            filled_index = _transform_striped(value, delta=delta, start=start, stop=stop)

        if filled_index is not None:
            return (filled_index + output_index, 1.0)

        output_index += _stripe_lines(stripe)

    return output_index, value


def _reverse_bound_and_stripes(data, stripes):
    idx, val = data
    ofs = 0

    for stripe in stripes:
        count = _stripe_lines(stripe)
        delta, start, stop = stripe
        if delta is None:
            if ofs == idx:
                return start
        else:
            if ofs <= idx < ofs+count:
                return start + delta * (idx - ofs)
        ofs += count
    return val


def _transform_35(value):
    # resulting value is in range [-11..11]
    int_val = np.round(value * 10.0)
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
    0: _transform_stripes_func(0),
    1: _split_bound_func(0.0),
    2: _split_bound_func(0.0),
    3: _split_bound_func(0.0),
    4: _split_bound_func(0.0),
    5: _transform_stripes_func(5),
    6: _transform_stripes_func(6),
    7: _transform_stripes_func(7),
    8: _transform_stripes_func(8),
    9: _transform_stripes_func(9),
    10: _transform_stripes_func(10),
    11: _transform_stripes_func(11),
    12: _transform_stripes_func(12),
    23: _transform_stripes_func(23),
    24: _transform_stripes_func(24),
    35: _transform_35
}


def _transform_striped(value, delta, start, stop):
    """
    Perform striped decoding of value
    :param value: value to decode
    :param delta: delta step for stripe
    :param start: first stripe
    :param stop: last stripe
    :return: None of one-hot index
    """
    h_d = delta / 2.0
    if value < start - h_d:
        return None
    if value > stop + h_d:
        return None
    ofs = 0
    bound = start + h_d
    while bound < stop + delta:
        if value < bound:
            return ofs
        ofs += 1
        bound += delta
    return None


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


def _reverse_35(data):
    idx, _ = data
    return float(idx-11) / 10.0


def _reverse_stripe_func(feature):
    def _reverse(data):
        return _reverse_bound_and_stripes(data, stripes=stripes[feature])
    return _reverse


reverse_transforms = {
    0: _reverse_stripe_func(0),
    1: _unsplit_bound,
    2: _unsplit_bound,
    3: _unsplit_bound,
    4: _unsplit_bound,
    5: _reverse_stripe_func(5),
    6: _reverse_stripe_func(6),
    7: _reverse_stripe_func(7),
    8: _reverse_stripe_func(8),
    9: _reverse_stripe_func(9),
    10: _reverse_stripe_func(10),
    11: _reverse_stripe_func(11),
    12: _reverse_stripe_func(12),
    23: _reverse_stripe_func(23),
    24: _reverse_stripe_func(24),
    35: _reverse_35,
}


def to_dense(sparse):
    res = np.zeros((RESULT_N_FEATURES,))
    apply_dense(res, sparse)
    return res

def apply_dense(vector, sparse):
    vector[sparse[0]] = sparse[1]
    return vector


def reverse(index, values):
    res = []
    ofs = 0

    for idx, data in enumerate(zip(index, values)):
        if idx in reverse_transforms:
            res.append(reverse_transforms[idx]((data[0]-ofs, data[1])))
            ofs += sizes[idx]
        else:
            res.append(data[1])
            ofs += 1

    return res