from search import random_multiple_tries
import numpy as np
from generator.objective import evaluate_constraints
from generator.objective import real_effect
from utils import alpha_cost, alpha_fair, cobyla_count


def random_search(dataset, t=60):
    """
    Random search method to find groups
    Args:
        dataset:
            Dataset configuration
        t: Integer
            Time budget (s)
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

    print("Start Random Search...")

    while 1 == 1:
        s1 = random_multiple_tries(
            X=X,
            y=y,
            model=model,
            dataset=dataset,
            alpha_cost=alpha_cost,
            alpha_fair=alpha_fair,
            count=cobyla_count,
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

        if sum(time_method) > t:
            break

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
