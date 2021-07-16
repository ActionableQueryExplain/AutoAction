from ._eff import effect
from ._cost import cost
from ._fair import fairness
from ._con import constraint
from ._realeff import real_effect


def objective(
    X_g,
    y,
    x,
    model,
    alpha_cost,
    alpha_fair,
    beta,
    fairness_constraint,
    action_constraints,
    instance_constraints,
    appro_instance_constraints,
    conditional_constraints,
    appro_conditional_constraints,
    l,
):
    """
    Objective Function
    Args:
        X_g, X, y: np.array
            Input Data
        model:
            Prediction model fit on data
        l: Integer
            Dimension of candidate action
        alpha_cost, alpha_fair: np.float
            Hyper-Parameters for objective function
        beta:
            Hyper-Parameter for penalty on constraints
        fairness_constraint: List
            Input for fairness information
        action_constraints, instance_constraints,
        appro_instance_constraints, conditional_constraints,
        appro_conditional_constraints: List
            Input constraints
    """

    """
    effectiveness score
    """
    effect_score = effect(X_g, y, x, model, alpha_cost, l)

    """
    cost score
    """
    cost_score = cost(X_g, x, alpha_cost)

    """
    fairness score
    """
    fairness_score = fairness(X_g, y, fairness_constraint, alpha_fair)

    """
    constraint score
    """
    constraint_score = constraint(
        X_g,
        x,
        action_constraints,
        instance_constraints,
        appro_instance_constraints,
        conditional_constraints,
        appro_conditional_constraints,
        beta,
        l,
    )

    """
    real effectiveness score
    """
    val = effect_score + cost_score + fairness_score + constraint_score

    return val, effect_score, cost_score, fairness_score, constraint_score


def objective_with_real(
    X_g,
    y,
    x,
    model,
    alpha_cost,
    alpha_fair,
    beta,
    fairness_constraint,
    instance_constraints,
    action_constraints,
    conditional_constraints,
    l,
):
    """
    Objective Function
    Args:
        X_g, X, y: np.array
            Input Data
        model:
            Prediction model fit on data
        l: Integer
            Dimension of candidate action
        alpha_cost, alpha_fair: np.float
            Hyper-Parameters for objective function
        beta:
            Hyper-Parameter for penalty on constraints
        fairness_constraint: List
            Input for fairness information
        action_constraints, instance_constraints,
        appro_instance_constraints, conditional_constraints,
        appro_conditional_constraints: List
            Input constraints
    """

    """
    effectiveness score
    """
    effect_score = effect(X_g, y, x, model, alpha_cost, l)

    """
    cost score
    """
    cost_score = cost(X_g, x, alpha_cost)

    """
    fairness score
    """
    fairness_score = fairness(X_g, y, fairness_constraint, alpha_fair)
    # fairness_score = real_fairness(X_g, y, x, fairness_constraint, alpha_fair, l)

    """
    constraint score
    """
    constraint_score = constraint(
        X_g,
        x,
        action_constraints,
        instance_constraints,
        conditional_constraints,
        beta,
        l,
    )

    """
    real effectiveness score
    """
    real_effect_score = real_effect(X_g, x, y, alpha_cost, l)

    val = effect_score + cost_score + fairness_score + constraint_score

    real_val = real_effect_score + cost_score + fairness_score + constraint_score

    return real_val, val, effect_score, cost_score, fairness_score, constraint_score
