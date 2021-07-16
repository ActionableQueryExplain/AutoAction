#!/usr/bin/env python3
from sklearn.ensemble import RandomForestClassifier

"""
Fit a model for given table.
"""
MODEL = RandomForestClassifier(max_depth=9, n_estimators=500)
