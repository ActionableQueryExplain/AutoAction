#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import sys
import itertools
import random
import time

import utils
from generator import action_generator
from generator.evaluator import action_evaluator

sys.path.append("../")

eta = utils.ETA
bound = utils.BOUND
max_iter = utils.MAX_ITER
num_rand_experiments = utils.NUM_RAND_EXPERIMENTS


class budget_based_search:
    """
    Hyperband Implementation
    https://arxiv.org/pdf/1603.06560.pdf
    Args:
        X, y: np.array
            Input Data
        model:
            Prediction model fit on data
        dataset:
            Dataset configuration
        iteration: Integer
            Budget
        discard_rate: float
            discard rate for groups
        alpha:
            Parameter to tune the number of iterations
        count:
            Number to execute cobyla
        alpha_cost, alpha_fair: np.float
            Hyper-Parameters for objective function
    """

    def __init__(
        self,
        X,
        y,
        model,
        dataset,
        iterations,
        discard_rate,
        alpha,
        count,
        alpha_cost,
        alpha_fair,
    ):
        self.X = X
        self.y = y
        self.model = model
        self.data = dataset.data
        self.dataset = dataset
        self.count = count
        self.iterations = iterations
        self.discard_rate = discard_rate
        self.alpha = alpha
        self.alpha_cost = alpha_cost
        self.alpha_fair = alpha_fair

    def run(self):

        best_group = []
        reg = self.model
        data = self.data
        group_attrs = self.dataset.group_attrs
        actionable_attrs = self.dataset.actionable_attrs
        fit_attrs = self.dataset.fit_attrs
        agg_attr = self.dataset.agg_attr
        iterations = self.iterations
        discard_rate = self.discard_rate
        alpha = self.alpha
        action_constraints = self.dataset.action_constraints
        instance_constraints = self.dataset.instance_constraints
        appro_instance_constraints = self.dataset.appro_instance_constraints
        conditional_constraints = self.dataset.conditional_constraints
        appro_conditional_constraints = self.dataset.appro_conditional_constraints

        uniques = []
        for attr in group_attrs:
            uniques.append(sorted(data[attr].unique()))
        groups = list(itertools.product(*uniques))

        start = time.time()

        configurations = groups.copy()

        logeta = lambda x: math.log(x) / math.log(discard_rate)
        s_max = math.floor(logeta(iterations))
        B = (s_max + 1) * iterations

        counter = 0
        best_loss = np.inf

        for s in reversed(range(s_max + 1)):

            # initial number of configurations
            n = int(math.ceil((B * (discard_rate ** s)) / (iterations * (s + 1))))

            # initial number of iterations per config
            r = iterations * discard_rate ** (-s)

            # n random configurations
            T = random.sample(configurations, min(n, len(configurations)))

            for i in range(s + 1):

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                # re-initalize parameters for same group

                n_configs = math.floor(n * discard_rate ** (-i))
                n_iterations = r * discard_rate ** (i)
                n_iterations = alpha * n_iterations

                val_losses = []

                for group in T:
                    group_str = ""
                    for attr in group:
                        group_str = group_str + str(attr)
                    counter = counter + 1

                    data_1 = data.copy()
                    for i in range(len(group)):
                        data_1 = data_1.loc[data_1[group_attrs[i]] == group[i]]

                    # filter non-existing groups
                    if len(data_1) < 1:
                        continue
                    X_g = data_1[fit_attrs]
                    y_g = data_1[agg_attr]
                    X_g = np.array(X_g)
                    y_g = np.array(y_g)
                    X_g = X_g.astype(np.float)
                    y_g = y_g.astype(np.float)
                    l = len(actionable_attrs)
                    generator = action_generator(
                        X_g,
                        y_g,
                        reg,
                        l,
                        self.alpha_cost,
                        self.alpha_fair,
                        count=self.count,
                    )

                    generator.add_constraint_fairness(
                        len(data),
                        float(np.array(data[agg_attr]).astype(np.float).mean()),
                        len(X_g),
                    )

                    for action_constraint in action_constraints:
                        generator.add_action_constraint(action_constraint)

                    for instance_constraint in instance_constraints:
                        generator.add_instance_constraint(instance_constraint)

                    for appro_instance_constraint in appro_instance_constraints:
                        generator.add_appro_instance_constraint(
                            appro_instance_constraint
                        )

                    for conditional_constraint in conditional_constraints:
                        generator.add_conditional_constraint(conditional_constraint)

                    for appro_conditional_constraint in appro_conditional_constraints:
                        generator.add_appro_conditional_constraint(
                            appro_conditional_constraint
                        )

                    (
                        theta,
                        val,
                        effective_val,
                        cost_val,
                        fairness_val,
                        constraint_val,
                    ) = generator.generate_best_action(
                        X_g, eta=eta, bound=bound, max_iter=int(n_iterations)
                    )

                    val_losses.append(val)

                    #    group_dict[group_str] = val
                    # else:
                    #    score = group_dict[group_str]
                    #    val_losses.append(score)

                    if val < best_loss:
                        best_loss = val
                        best_group = group

                    # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices]
                T = T[0 : int(n_configs / eta)]

        data_1 = data.copy()
        for i in range(len(best_group)):
            data_1 = data_1.loc[data_1[group_attrs[i]] == best_group[i]]
        X_g = data_1[fit_attrs]
        y_g = data_1[agg_attr]
        X_g = np.array(X_g)
        y_g = np.array(y_g)
        X_g = X_g.astype(np.float)
        y_g = y_g.astype(np.float)
        l = len(actionable_attrs)
        generator = action_generator(
            X_g, y_g, reg, l, self.alpha_cost, self.alpha_fair, self.count
        )

        generator.add_constraint_fairness(
            len(data), float(np.array(data[agg_attr]).astype(np.float).mean()), len(X_g)
        )
        for action_constraint in action_constraints:
            generator.add_action_constraint(action_constraint)

        for instance_constraint in instance_constraints:
            generator.add_instance_constraint(instance_constraint)

        for appro_conditional_constraint in appro_conditional_constraints:
            generator.add_appro_conditional_constraint(appro_conditional_constraint)

        (
            theta,
            estimated_score,
            effective_score,
            cost_score,
            fairness_score,
            constraint_score,
        ) = generator.generate_best_action(X_g, eta=eta, bound=bound, max_iter=max_iter)

        end = time.time()

        return (
            theta,
            best_group,
            end - start,
            estimated_score,
            effective_score,
            cost_score,
            fairness_score,
            constraint_score,
        )
