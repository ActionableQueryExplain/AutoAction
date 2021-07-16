from baselines import hyperband
from dataConfig import dataLoader
import numpy as np


class sround_hyperband:
    """
    The single round Hyperband.
    Args:
        ds_config: String
            Dataset used
        trials: Integer
            Number of trials
        R: Integer
            Budget parameter
    """

    def __init__(self, ds_config, trials, R):

        self.ds_config = ds_config
        self.trials = trials
        self.R = R

    def run(self):
        dl = dataLoader()
        dataset = dl.loader(dataset=self.ds_config)

        all_scores = []
        all_times = []

        for index in range(self.trials):
            actions, groups, scores, times = hyperband(dataset=dataset, R=self.R)

            all_scores.append(scores[0])
            all_times.append(times[0])

        print(
            "The average time under budget "
            + str(self.R)
            + " for single round Hyperband is "
            + str(np.mean(all_times))
        )
        print(
            "The average score under budget "
            + str(self.R)
            + " for single round Hyperband is "
            + str(np.mean(all_scores))
        )
