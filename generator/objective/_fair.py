def fairness(X_g, y, fairness_constraint, alpha_fair):
    """
    Fairness
    Args:
        X_g, y: np.array
            Input Data
        fairness_constraint: List
            Input for fairness information
        alpha_fair: np.float
            Hyper-Parameters for objective function
    """
    fairness_constraint = fairness_constraint
    m = X_g.shape[0]
    y = y.reshape((m, 1))
    f = 0
    if len(fairness_constraint) > 0:
        f = -(fairness_constraint[1] - y.mean())

    return alpha_fair * f
