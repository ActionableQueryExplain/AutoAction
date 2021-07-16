import numpy as np
from postprocessor import interpreter


def suggester(actions, groups, scores, group_attrs, actionable_attrs):
    """
    Suggest the best solution so far
    Args:
        actions: List
            candidate actions
        groups: List
            candidate groups
        score: List
            candidate scores
        group_attrs: List
            group attributes
        actionable_attrs: List
            actionable attributes
    """
    score = 0
    best_action = []
    best_group = []
    for i in range(len(scores)):
        if scores[i] < score:
            score = scores[i]
            best_action = actions[i]
            best_group = groups[i]

    if score == 0:
        print("No applicable action is generated under the time budget.")
    else:
        print("The best action generated so far is suggested: ")
        interpreter(group_attrs, actionable_attrs, best_action, best_group)
