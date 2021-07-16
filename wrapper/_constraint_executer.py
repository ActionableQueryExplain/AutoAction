from autoaction import autoAction
from dataConfig import dataLoader
import numpy as np


class constraint_executer:
    """
    Experiments for supporting constraints.
    Args:
        ds_config: String
            Dataset used
        trials: Integer
            Number of trials
        constraint_setting: String
            Constraint setting
    """

    def __init__(self, ds_config, trials, times, constraint_setting="default"):

        self.ds_config = ds_config
        self.trials = trials
        self.times = times
        self.constraint_setting = constraint_setting

    def run(self):
        dl = dataLoader()
        dataset = dl.loader(
            dataset=self.ds_config, constraint_setting=self.constraint_setting
        )

        all_scores_b = []
        all_scores_m = []
        for i in range(len(self.times)):
            all_scores_b.append([])
            all_scores_m.append([])

        for index in range(self.trials):
            actions_b, groups_b, scores_b, times_b = autoAction(
                dataset=dataset, t=self.times[-1], constraint_setup="post_verify"
            )

            for i in range(len(self.times)):
                t = self.times[i]
                score = 0
                for j in range(len(scores_b)):
                    s = scores_b[j]
                    if times_b[j] <= t:
                        score = min(score, s)
                all_scores_b[i].append(score)

            actions_m, groups_m, scores_m, times_m = autoAction(
                dataset=dataset, t=self.times[-1]
            )

            for i in range(len(self.times)):
                t = self.times[i]
                score = 0
                for j in range(len(scores_m)):
                    s = scores_m[j]
                    if times_m[j] <= t:
                        score = min(score, s)
                all_scores_m[i].append(score)

        for i in range(len(self.times)):
            print(
                "The average score under "
                + str(self.times[i])
                + "s for post verify is "
                + str(np.mean(all_scores_b[i]))
            )

        print("----------------------------------")
        for i in range(len(self.times)):
            print(
                "The average score under "
                + str(self.times[i])
                + "s for autoaction is "
                + str(np.mean(all_scores_m[i]))
            )
