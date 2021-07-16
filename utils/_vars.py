#!/usr/bin/env python3

"""
Hyper-Parameters for Experiments
"""

# Hyper-Parameters for COBYLA
ETA = 1e-1
BOUND = 1e-3
MAX_ITER = 200
NUM_RAND_EXPERIMENTS = 1

# Hyper-Parameters for Hyperband
R = 10  # budget
mu = 5  # discard rate
alpha = 1
cobyla_count = 3  # times to run cobyla

# Hyper-Parameters for objective function
alpha_cost = 0.99
alpha_fair = 20
