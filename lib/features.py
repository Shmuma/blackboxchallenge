import numpy as np
import array

ORIGIN_N_FEATURES = 36
RESULT_N_FEATURES = 37


def transform(state, time):
    """
    All features except last transfomed into bits value, last left as is
    :param state:
    :return: transformed numpy vector
    """
    return np.append(state, time)
    # arr = array.array('f', state[:-1])
    # last = int(state[-1] * 10.0) + 11
    #
    # a_ints = array.array('I')
    # a_ints.fromstring(arr.tostring())
    # v = map(lambda v: np.binary_repr(v, width=32), a_ints)
    # v = map(int, "".join(v))
    # v.append(last)
    # return np.array(v, dtype=np.int8)


data = [-0.73334587, -0.93910491, -0.94270045, -1.36341822, -1.70008028, -0.65676075,
        -0.66000223, -0.61031222, -0.61547941, -0.73753935, -0.74450946, -2.11130285,
        -0.040695058, -1.24927127,  0.90754294, -0.21203995,  0.3009333 , -0.49948287,
        -0.646667,   -0.31782657,  0.25072268,  1.20941901,  0.24673782, -0.2735582 ,
        0.78459948,   0.47483093,  0.15551165, -0.09912075, -0.3076221 , -0.46290749,
        0.112813171,  -0.4509145,  1.5137006,   -0.99996543,  0.87362069,  0.
        ]

def bench():

    import timeit
    print timeit.timeit('features.transform(features.data)', setup='import features', number=10000)


if __name__ == "__main__":
    bench()
