from search import budget_based_search
import numpy as np
from generator.objective import evaluate_constraints
from generator.objective import real_effect
from utils import mu, alpha, cobyla_count, alpha_cost, alpha_fair


def hyperband(dataset, R=20):
    """
    Single round Hyperband
    Args:
        dataset:
            Dataset configuration
        R: Integer
            Input budget
    """
    group_attrs = dataset.group_attrs
    other_attrs = dataset.other_attrs
    actionable_attrs = dataset.actionable_attrs
    agg_attr = dataset.agg_attr
    fit_attrs = dataset.fit_attrs
    data = dataset.data
    X, y, model = dataset.X, dataset.y, dataset.model
    action_constraints = dataset.action_constraints
    instance_constraints = dataset.instance_constraints
    appro_instance_constraints = dataset.appro_instance_constraints
    conditional_constraints = dataset.conditional_constraints
    appro_conditional_constraints = dataset.appro_conditional_constraints

    time_method = []
    score_method = []
    action_method = []
    group_method = []
    total = 0

    print("Start Single Round Hyperband...")

    s1 = budget_based_search(
        X=X,
        y=y,
        model=model,
        dataset=dataset,
        iterations=R,
        discard_rate=mu,
        alpha=alpha,
        count=cobyla_count,
        alpha_cost=alpha_cost,
        alpha_fair=alpha_fair,
    )
    (
        theta,
        best_group,
        running_time,
        estimated_score,
        effective_score,
        cost_score,
        fairness_score,
        constraint_score,
    ) = s1.run()

    result = evaluate_constraints(
        X_g=X,
        X_global=data,
        x=theta,
        group=best_group,
        group_attrs=group_attrs,
        actionable_attrs=actionable_attrs,
        fit_attrs=fit_attrs,
        agg_attr=agg_attr,
        action_constraints=action_constraints,
        instance_constraints=instance_constraints,
        appro_instance_constraints=appro_instance_constraints,
        conditional_constraints=conditional_constraints,
        appro_conditional_constraints=appro_conditional_constraints,
    )
    total += running_time

    if result:
        action_method.append(theta)
        group_method.append(best_group)
        score_method.append(estimated_score)
        time_method.append(total)
    else:
        action_method.append([])
        group_method.append([])
        score_method.append(0)
        time_method.append(total)

    return action_method, group_method, score_method, time_method
