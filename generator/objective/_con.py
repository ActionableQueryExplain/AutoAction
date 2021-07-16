import numpy as np
import math


def constraint(
    X_g,
    x,
    action_constraints,
    instance_constraints,
    appro_instance_constraints,
    conditional_constraints,
    appro_conditional_constraints,
    beta,
    l,
):
    """
    Penalty on violation of constraints
    Args:
        X_g, x: np.array
            Input Data
        action_constraints, instance_constraints,
        appro_instance_constraints, conditional_constraints,
        appro_conditional_constraints: List
            Input constraints
        beta:
            Hyper-Parameter for penalty on constraints
        l: Integer
            Dimension of candidate action
    """
    m = X_g.shape[0]
    n = X_g.shape[1]

    A = np.ones((m, 1))

    x_0 = np.zeros((1, n - l))
    x_1 = x.reshape((1, l))
    x = np.concatenate((x_0, x_1), axis=1)

    """
    action based constraints
    """
    action_penalty = 0

    if len(action_constraints) > 0:
        for action_constraint in action_constraints:
            penalty = 0
            for i in range(l):
                penalty += action_constraint[2 * i] * (
                    math.pow(x_1[0][-l + i], action_constraint[2 * i + 1])
                )
            penalty -= action_constraint[-1]
            relu1 = lambda t: min(max(0, t), 1)
            vfunc = np.vectorize(relu1)
            penalty = vfunc(penalty)
            action_penalty += penalty.mean()

    """
    instance based constraints
    """
    instance_penalty = 0

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
                relu1 = lambda t: min(max(0, t), 1)
                vfunc = np.vectorize(relu1)
                p = vfunc(p)
                penalty += p.mean()
            instance_penalty += penalty

    """
    conditional based constraints
    """
    conditional_penalty = 0

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
                if p1 > 0:
                    relu1 = lambda t: min(max(0, t), 1)
                    vfunc = np.vectorize(relu1)
                    p2 = vfunc(-p2)
                    penalty += p2.mean()
            conditional_penalty += penalty

    """
    approximate constraints
    """

    appro_instance_penalty = 0
    appro_conditional_penalty = 0
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
        appro_instance_penalty += max(0, ((s - a + 0.0) / len(instances)) - 0.01)

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

        appro_conditional_penalty += max(0, ((s - a + 0.0) / len(instances)) - 0.01)

    return beta * (
        action_penalty
        + instance_penalty
        + conditional_penalty
        + appro_instance_penalty
        + appro_conditional_penalty
    )
