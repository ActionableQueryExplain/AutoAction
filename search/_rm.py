#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import sys
import os
import time

from generator import action_generator
from generator.evaluator import action_evaluator

import itertools
import random

from utils import ETA, BOUND, MAX_ITER, NUM_RAND_EXPERIMENTS
import numpy as np

sys.path.append(os.path.abspath("../utils"))

eta = ETA
bound = BOUND
max_iter = MAX_ITER
num_rand_experiments = NUM_RAND_EXPERIMENTS


class random_multiple_tries:
    """
    Random Search Method.
    Args:
        X, y: np.array
            Input Data
        model:
            Prediction model fit on data
        dataset:
            Dataset configuration
        alpha_cost, alpha_fair: np.float
            Hyper-Parameters for objective function
        count: Integer
            Count to run cobyla
    """

    def __init__(self, X, y, model, dataset, alpha_cost, alpha_fair, count):
        self.X = X
        self.y = y
        self.model = model
        self.data = dataset.data
        self.dataset = dataset
        self.alpha_cost = alpha_cost
        self.alpha_fair = alpha_fair
        self.count = count

    def run(self):

        # naive: random pick multiple grouo out of all groups until the time exceeds limit

        best_estimated_score = np.inf
        best_group = []

        best_group = []
        reg = self.model
        data = self.data
        group_attrs = self.dataset.group_attrs
        actionable_attrs = self.dataset.actionable_attrs
        fit_attrs = self.dataset.fit_attrs
        agg_attr = self.dataset.agg_attr
        action_constraints = self.dataset.action_constraints
        instance_constraints = self.dataset.instance_constraints
        appro_instance_constraints = self.dataset.appro_instance_constraints
        conditional_constraints = self.dataset.conditional_constraints
        appro_conditional_constraints = self.dataset.appro_conditional_constraints

        uniques = []
        for attr in group_attrs:
            uniques.append(sorted(data[attr].unique()))
        groups = list(itertools.product(*uniques))
        temp_groups = groups.copy()

        t = 0
        start = time.time()
        best_group = random.choice(temp_groups)
        while 1 == 1:
            best_group = random.choice(temp_groups)
            data_1 = data.copy()
            for i in range(len(best_group)):
                data_1 = data_1.loc[data_1[group_attrs[i]] == best_group[i]]
            if len(data_1) < 1:
                continue
            break

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

        for appro_instance_constraint in appro_instance_constraints:
            generator.add_instance_constraint(appro_instance_constraint)

        for conditional_constraint in conditional_constraints:
            generator.add_appro_conditional_constraint(conditional_constraint)

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
