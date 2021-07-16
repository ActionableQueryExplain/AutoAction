import numpy as np


def normalized_sigmoid_fkt(a, b, x):

    """
    Returns array of a horizontal mirrored normalized sigmoid function
    output between 0 and 1
    Function parameters a = center; b = width
    """

    s = 1 / (1 + np.exp(b * (x - a)))
    return s
