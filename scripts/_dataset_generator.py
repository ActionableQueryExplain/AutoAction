#!/usr/bin/env python3
import random
import numpy as np
import pandas as pd
import sys


def dataset_generator(size, group1, group2):

    """
    Generate synthetic datasets for a credit card application scearino
    Refer to https://arxiv.org/pdf/2002.06278.pdf
    Args:
        size: integer
            Total number of rows of synthetic dataset, 100K and 1M for the experiments.
        group1: integer
            Total number of groups for variable X1.
        group2: integer
            Total number of groups for variable X2.

    Execution: ./scripts/_dataset_generator.py size group1 group2 outputPath
    # The output should be a csv file
    """

    def constrained_sum_sample_pos(n, total):
        """
        Randomly separate the population to n groups.
        """
        dividers = sorted(random.sample(range(1, total), n - 1))
        return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

    dataset = []
    sizes = constrained_sum_sample_pos(group1 * group2, size)
    n_groups = group1 * group2
    map_1 = dict()
    map_2 = dict()
    for i in range(group1):
        map_1[i] = random.uniform(0, 1)
    for i in range(group2):
        map_2[i] = random.uniform(0, 1) - 0.5

    for i in range(len(sizes)):
        s = sizes[i]
        for j in range(s):
            g1 = int(i % group1)
            g2 = int(i / group1)

            u1 = np.random.poisson(20 * map_1[g1])
            u2 = 0.25 * np.random.normal(map_2[g2], 1)

            x1 = u1
            x2 = 0.3 * x1 + u2

            t = x1 + 5 * x2 - 22.5
            y = 0 if t <= 0 else 1

            data = [g1, g2, u1, u2, x1, x2, y]
            dataset.append(data)

    df = pd.DataFrame(dataset)
    return df


if __name__ == "__main__":
    size = int(sys.argv[1])
    group1 = int(sys.argv[2])
    group2 = int(sys.argv[3])
    path = sys.argv[4]
    df = dataset_generator(size=size, group1=group1, group2=group2)
    df.to_csv(path)
    print("dataset generated.")
