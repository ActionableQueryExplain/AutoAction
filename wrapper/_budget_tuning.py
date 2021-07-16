from autoaction import autoAction
from dataConfig import dataLoader
import numpy as np


class budget_tuner:
    """
    Tune the Budget R for AutoAction.
    Args:
        R: Integer
            The input budget.
        ds_config: String
            Dataset used
        trials: Integer
            Number of trials
        time: Integer
            Time budget(s)
    """

    def __init__(self, R, ds_config, trials, time):

        self.R = R
        self.ds_config = ds_config
        self.trials = trials
        self.time = time

    def run(self):
        dl = dataLoader()
        dataset = dl.loader(dataset=self.ds_config)

        all_scores = []

        for i in range(self.trials):
            actions, groups, scores, times = autoAction(
                dataset=dataset, t=self.time, R=self.R
            )
            score = 0
            for s in scores:
                score = min(score, s)
            all_scores.append(score)

        print(
            "The average score under "
            + str(self.time)
            + "s is "
            + str(np.mean(all_scores))
        )
