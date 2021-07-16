import numpy as np
import math


def cost(X_g, x, alpha_cost):
    """
    Cost
    Args:
        X_g, x: np.array
            Input Data
        alpha_cost: np.float
            Hyper-Parameters for objective function
    """

    alpha = alpha_cost
    c = np.linalg.norm(x, ord=2) * len(X_g)

    return (1 - alpha) * c
