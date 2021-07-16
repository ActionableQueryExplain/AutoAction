import numpy as np


def effect(X_g, y, x, model, alpha_cost, l):
    """
    Effect
    Args:
        X_g, x, y: np.array
            Input Data
        model:
            Prediction model fit on data
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

    B_ = model.predict(np.dot(A, x) + X_g).reshape((m, 1))
    for i in range(len(y)):
        if y[i][0] == 0 and B_[i][0] == 1:
            S = np.dot(A, x) + X_g

    f = lambda x: min(0, x.any())

    alpha = alpha_cost
    return alpha * min(np.sum(y) - np.sum(B_), 0)  # alpha*np.sum(y - B_)
