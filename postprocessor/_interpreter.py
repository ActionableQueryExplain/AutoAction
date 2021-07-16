import numpy as np


def interpreter(group_attrs, actionable_attrs, theta, group):
    """
    Interprete generated actions
    Args:
        group_attrs: List
            group attributes
        actionable_attrs: List
            actionable attributes
        theta: np.array
            Candidate action
        group: List
            Candidate group
    """

    s = ""
    for i in range(len(group)):
        s += group_attrs[i] + " is " + group[i]
        if i != len(group) - 1:
            s += ", "

    print("For the group, " + s)

    s = ""
    for i in range(len(actionable_attrs)):
        if theta[0][i] >= 0:
            s += (
                "increase " + actionable_attrs[i] + " by " + format(theta[0][i], ".2f")
            )  # str(theta[0][i])
        else:
            s += (
                "decrease " + actionable_attrs[i] + " by " + format(theta[0][i], ".2f")
            )  # str(-theta[0][i])
        if i != len(actionable_attrs) - 1:
            s += ", "

    print("Take the action, " + s)
