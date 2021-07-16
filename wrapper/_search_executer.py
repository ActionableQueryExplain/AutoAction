from autoaction import autoAction
from dataConfig import dataLoader
import numpy as np
from baselines import random_search


class search_executer:
    """
    Perform the search and comparision between methods.
    Args:
        ds_config: String
            Dataset used
        trials: Integer
            Number of trials
        time: Integer
            Time budget(s)
    """

    def __init__(self, ds_config, trials, times):

        self.ds_config = ds_config
        self.trials = trials
        self.times = times

    def run(self):
        dl = dataLoader()
        dataset = dl.loader(dataset=self.ds_config)

        all_scores_b = []
        all_scores_m = []
        for i in range(len(self.times)):
            all_scores_b.append([])
            all_scores_m.append([])

        for index in range(self.trials):
            actions_b, groups_b, scores_b, times_b = random_search(
                dataset=dataset, t=self.times[-1]
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
                + "s for random search is "
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
