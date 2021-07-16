import numpy as np
import math


def evaluate_constraints(
    X_g,
    X_global,
    x,
    group,
    group_attrs,
    actionable_attrs,
    fit_attrs,
    agg_attr,
    action_constraints,
    instance_constraints,
    appro_instance_constraints,
    conditional_constraints,
    appro_conditional_constraints,
):
    """
    Evaluate whether an action observes constraints
    Args:
        X_g: np.array
            Input Data
        x: np.array:
            Input action
        group: List
            Input group
        group_attrs: List
            Group attributes
        actionable_attrs: List
            Actionable attributes
        fit_attrs: List
            Attributes to fit the model
        agg_attr: List
            Aggregation attribute
        action_constraints, instance_constraints,
        appro_instance_constraints, conditional_constraints,
        appro_conditional_constraints: List
            Input constraints
    """

    data = X_global
    data_1 = data.copy()
    for i in range(len(group)):
        data_1 = data_1.loc[data_1[group_attrs[i]] == group[i]]
    X_g = data_1[fit_attrs]
    y_g = data_1[agg_attr]
    X_g = np.array(X_g)
    y_g = np.array(y_g)
    X_g = X_g.astype(np.float)
    y_g = y_g.astype(np.float)
    l = len(actionable_attrs)

    m = X_g.shape[0]
    n = X_g.shape[1]

    A = np.ones((m, 1))

    x_0 = np.zeros((1, n - l))
    x_1 = x.reshape((1, l))
    x = np.concatenate((x_0, x_1), axis=1)

    """
    action based constraints
    """

    if len(action_constraints) > 0:
        for action_constraint in action_constraints:
            penalty = 0
            for i in range(l):
                penalty += action_constraint[2 * i] * (
                    math.pow(x_1[0][-l + i], action_constraint[2 * i + 1])
                )
            penalty -= action_constraint[-1]
            if penalty > 0:
                return False

    """
    instance based constraints
    """

    if len(instance_constraints) > 0:
        for instance_constraint in instance_constraints:
            penalty = 0
            for t in X_g:
                p = 0
                for i in range(l):
                    p += instance_constraint[2 * i] * (
                        math.pow(t[-l + i], instance_constraint[2 * i + 1])
                    )
                p -= instance_constraint[-1]
                if p > 0:
                    return False

    """
    conditional based constraints
    """

    if len(conditional_constraints) > 0:
        for conditional_constraint in conditional_constraints:
            penalty = 0
            for t in X_g:
                p1 = 0
                p2 = 0
                for i in range(l):
                    if conditional_constraint[2 * i] == 0:
                        p1 += conditional_constraint[2 * i + 1] * t[-l + i]
                    else:
                        p2 += conditional_constraint[2 * i + 1] * t[-l + i]

                p1 -= conditional_constraint[-2]
                p2 -= conditional_constraint[-1]
                if p1 > 0 and p2 < 0:
                    return False

    """
    approximate constraints
    """

    instances = np.dot(A, x) + X_g

    for appro_instance_constraint in appro_instance_constraints:
        a = 0
        s = 0
        for t in X_g:
            p = 0
            for i in range(l):
                p += appro_instance_constraint[2 * i] * (
                    math.pow(t[-l + i], appro_instance_constraint[2 * i + 1])
                )
            p -= appro_instance_constraint[-1]
            if p < 0:
                a += 1
        for t in instances:
            p = 0
            for i in range(l):
                p += appro_instance_constraint[2 * i] * (
                    math.pow(t[-l + i], appro_instance_constraint[2 * i + 1])
                )
            p -= appro_instance_constraint[-1]
            if p < 0:
                s += 1
        if ((s - a + 0.0) / len(instances)) - 0.01 > 0:
            return False

    for appro_conditional_constraint in appro_conditional_constraints:
        a = 0
        s = 0
        for t in X_g:
            p1 = 0
            p2 = 0
            for i in range(l):
                if appro_conditional_constraint[2 * i] == 0:
                    p1 += appro_conditional_constraint[2 * i + 1] * t[-l + i]
                else:
                    p2 += appro_conditional_constraint[2 * i + 1] * t[-l + i]

            p1 -= appro_conditional_constraint[-2]
            p2 -= appro_conditional_constraint[-1]
            if p2 < 0 and p1 > 0:
                a += 1
        for t in instances:
            p1 = 0
            p2 = 0
            for i in range(l):
                if appro_conditional_constraint[2 * i] == 0:
                    p1 += appro_conditional_constraint[2 * i + 1] * t[-l + i]
                else:
                    p2 += appro_conditional_constraint[2 * i + 1] * t[-l + i]

            p1 -= appro_conditional_constraint[-2]
            p2 -= appro_conditional_constraint[-1]

            if p2 < 0 and p1 > 0:
                s += 1

        if ((s - a + 0.0) / len(instances)) - 0.01 > 0:
            return False

    return True
