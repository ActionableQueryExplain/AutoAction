from utils import (
    ETA,
    BOUND,
    MAX_ITER,
    NUM_RAND_EXPERIMENTS,
    R,
    mu,
    alpha,
    cobyla_count,
    alpha_cost,
    alpha_fair,
)
from search import budget_based_search
from generator.objective import evaluate_constraints
import numpy as np
from generator.objective import real_effect
from postprocessor import interpreter, suggester

eta = ETA
bound = BOUND
max_iter = MAX_ITER
num_rand_experiments = NUM_RAND_EXPERIMENTS


def autoAction(dataset, t=60, constraint_setup="default", R=R, demo=False):
    """
    AutoAction implementation
    Args:
        dataset:
            Dataset configuration
        t: Integer
            Time budget (s)
        constraint_setup: String
            Setup for constraints
        demo: Boolean
            Whether showing demo or not
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

    if constraint_setup == "post_verify":
        dataset.action_constraints = []
        dataset.instance_constraints = []
        dataset.appro_instance_constraints = []
        dataset.conditional_constraints = []
        dataset.appro_conditional_constraints = []

    time_method = []
    score_method = []
    action_method = []
    group_method = []
    total = 0

    print("Start autoAction...")
    while 1 == 1:
        flag = 0
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
        if total > t:
            break

        """
        real score for generated dataset
        """
        if dataset.name == "credit100k" or dataset.name == "credit1m":
            data_2 = data.copy()
            for i in range(len(best_group)):
                data_2 = data_2.loc[data_2[group_attrs[i]] == best_group[i]]
            X_g_2 = data_2[fit_attrs]
            y_g_2 = data_2[agg_attr]
            X_g_2 = np.array(X_g_2)
            y_g_2 = np.array(y_g_2)
            X_g_2 = X_g_2.astype(np.float)
            y_g_2 = y_g_2.astype(np.float)
            l = len(actionable_attrs)
            effect = real_effect(X_g_2, theta, y_g_2, alpha_cost, l)
            estimated_score = effect + cost_score + fairness_score + constraint_score

        if result:
            action_method.append(theta)
            group_method.append(best_group)
            score_method.append(estimated_score)
            time_method.append(total)
            if demo:
                print("=================")
                print("action generated.")
                interpreter(group_attrs, actionable_attrs, theta, best_group)
                print("=================")
        else:
            action_method.append([])
            group_method.append([])
            score_method.append(0)
            time_method.append(total)

    if demo:
        suggester(
            action_method, group_method, score_method, group_attrs, actionable_attrs
        )

    return action_method, group_method, score_method, time_method
