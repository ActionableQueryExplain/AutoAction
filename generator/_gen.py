#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
import math
from .objective import objective
from generator.evaluator import action_evaluator
import time


class action_generator:
    """
    Generate actions given input dataframe and objective function
    Args:
        X, y: np.array
            Input Data
        model:
            Prediction model fit on data
        l: Integer
            Dimension of candidate action
        alpha_cost, alpha_fair: np.float
            Hyper-Parameters for objective function
        count:
            Number of run cobyla
        beta:
            Hyper-Parameter for penalty on constraints
    """

    def __init__(self, X, y, model, l, alpha_cost, alpha_fair, count, beta=1e8):
        self.X = X
        self.y = y
        self.model = model
        self.alpha_cost = alpha_cost
        self.alpha_fair = alpha_fair
        self.beta = beta
        self.fairness_constraint = []
        self.action_constraints = []
        self.instance_constraints = []
        self.appro_instance_constraints = []
        self.conditional_constraints = []
        self.appro_conditional_constraints = []
        self.l = l
        self.count = count

    # Add constraint to problem
    def add_constraint_fairness(self, population_size, population_avg, group_size):
        self.fairness_constraint = [population_size, population_avg, group_size]

    def add_action_constraint(self, action_constraint):
        self.action_constraints.append(action_constraint)

    def add_instance_constraint(self, instance_constraint):
        self.instance_constraints.append(instance_constraint)

    def add_appro_instance_constraint(self, appro_instance_constraint):
        self.appro_instance_constraints.append(appro_instance_constraint)

    def add_conditional_constraint(self, conditional_constraint):
        self.conditional_constraints.append(conditional_constraint)

    def add_appro_conditional_constraint(self, appro_conditional_constraint):
        self.appro_conditional_constraints.append(appro_conditional_constraint)

    # generate initial actions randomly
    def generate_initial_actions(self, X):
        theta = np.random.rand(1, self.l)

        return theta

    # Generate best action
    def generate_best_action(self, X_g, eta, bound, max_iter):

        theta = self.generate_initial_actions(X_g)
        model = self.model

        m = X_g.shape[0]
        n = X_g.shape[1]

        A = np.ones((m, 1))

        y = self.y.reshape((m, 1))

        def cost(x):
            return objective(
                X_g,
                y,
                x,
                model,
                self.alpha_cost,
                self.alpha_fair,
                self.beta,
                self.fairness_constraint,
                self.action_constraints,
                self.instance_constraints,
                self.appro_instance_constraints,
                self.conditional_constraints,
                self.appro_conditional_constraints,
                self.l,
            )[0]

        best_action = []
        best_score = np.inf
        best_effect = 0
        best_cost = 0
        best_fairness = 0
        best_constraint = 0
        for i in range(self.count):

            res = minimize(cost, theta, method="COBYLA", options={"maxiter": max_iter})
            action = res.x.reshape((1, self.l))

            evaluator = action_evaluator(X_g, self.y, action, model, self.l, self)

            (
                val,
                effective_val,
                cost_val,
                fairness_val,
                constraint_val,
            ) = evaluator.objective_score()
            if val < best_score:
                best_score = val
                best_action = action
                best_effect = effective_val
                best_cost = cost_val
                best_fairness = fairness_val
                best_constraint = constraint_val

        return (
            best_action,
            best_score,
            best_effect,
            best_cost,
            best_fairness,
            best_constraint,
        )
