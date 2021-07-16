#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..objective import objective, objective_with_real


class action_evaluator:
    """
    Evaluate actions
    Args:
        X, y: np.array
            Input Data
        action: np.array
            Candidate action
        model:
            Prediction model fit on data
        l: Integer
            Dimension of candidate action
        generator:
            Input action generator
    """

    def __init__(self, X, y, action, model, l, generator):

        global data_size
        self.X = X
        self.y = y
        self.action = action
        self.model = model
        self.l = l
        self.alpha_cost = generator.alpha_cost
        self.alpha_fair = generator.alpha_fair
        self.beta = generator.beta
        self.fairness_constraint = generator.fairness_constraint
        self.action_constraints = generator.action_constraints
        self.instance_constraints = generator.instance_constraints
        self.appro_instance_constraints = generator.appro_instance_constraints
        self.conditional_constraints = generator.conditional_constraints
        self.appro_conditional_constraints = generator.appro_conditional_constraints

    def objective_score(self):

        return objective(
            self.X,
            self.y,
            self.action,
            self.model,
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
        )

    def objective_score_with_real(self):

        return objective_with_real(
            self.X,
            self.y,
            self.action,
            self.model,
            self.alpha_cost,
            self.alpha_fair,
            self.beta,
            self.fairness_constraint,
            self.instance_constraints,
            self.action_constraints,
            self.conditional_constraints,
            self.l,
        )
