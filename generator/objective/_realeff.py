import numpy as np


def real_effect(X_g, x, y, alpha_cost, l):
    """
    Calculate real effect. Applies to credit datasets generated using _dataset_generator.py
    Args:
        X_g, X, y: np.array
            Input Data
        alpha_cost: np.float
            Hyper-Parameters for objective function
        l: Integer
            Dimension of candidate action
    """

    m = X_g.shape[0]
    n = X_g.shape[1]

    A = np.ones((m, 1))
    y = y.reshape((m, 1))

    x_0 = np.zeros((1, n - l))
    x_1 = x.reshape((1, l))
    x = np.concatenate((x_0, x_1), axis=1)
    alpha = alpha_cost

    B_real = np.zeros((m, 1))
    X_real = np.dot(A, x) + X_g

    for i in range(m):
        t = X_real[i][-2] + 5 * X_real[i][-1] - 22.5
        B_real[i][0] = 1 if t >= 0 else 0

    return alpha * min(np.sum(y) - np.sum(B_real), 0)
