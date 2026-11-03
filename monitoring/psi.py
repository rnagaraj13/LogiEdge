"""Population Stability Index for binned confidence-score distributions."""

from __future__ import annotations

import math

BIN_EDGES = (0.0, 0.25, 0.50, 0.75, 1.0)
EPSILON = 1e-6


def proportions(values):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("Cannot build a confidence distribution from no values.")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Confidence values must be in [0, 1].")
    counts = [0, 0, 0, 0]
    for value in values:
        counts[min(int(value * 4), 3)] += 1
    total = sum(counts)
    return [count / total for count in counts]


def calculate_psi(reference, current, epsilon=EPSILON):
    if len(reference) != 4 or len(current) != 4:
        raise ValueError("PSI distributions must contain exactly four bins.")
    if sum(reference) <= 0 or sum(current) <= 0:
        raise ValueError("PSI distributions must have positive total mass.")
    ref = [max(float(value) / sum(reference), epsilon) for value in reference]
    cur = [max(float(value) / sum(current), epsilon) for value in current]
    ref_sum, cur_sum = sum(ref), sum(cur)
    ref = [value / ref_sum for value in ref]
    cur = [value / cur_sum for value in cur]
    return sum((actual - expected) * math.log(actual / expected)
               for expected, actual in zip(ref, cur))
