"""Metric maths, checked against cases whose answers are known by hand."""

import numpy as np

from digitood.evaluation.metrics import (
    auroc, fpr_at_tpr, reject_threshold_high, reject_threshold_low,
)


def test_auroc_perfect_separation():
    assert auroc(np.array([0.0, 1, 2]), np.array([3.0, 4, 5])) == 1.0


def test_auroc_inverted_separation():
    assert auroc(np.array([3.0, 4, 5]), np.array([0.0, 1, 2])) == 0.0


def test_auroc_all_ties_is_a_coin_flip():
    # Ties must count as half, not as wins -- a saturated model emits many
    # identical scores and would otherwise report a flattering AUROC.
    assert auroc(np.ones(50), np.ones(50)) == 0.5


def test_auroc_of_identical_distributions_is_about_half():
    rng = np.random.default_rng(0)
    value = auroc(rng.normal(size=4000), rng.normal(size=4000))
    assert 0.47 < value < 0.53


def test_fpr_at_tpr_is_zero_when_separation_is_clean():
    assert fpr_at_tpr(np.zeros(100), np.ones(100), 0.95) == 0.0


def test_high_threshold_respects_the_false_reject_budget():
    """Energy-style score: reject above the threshold."""
    scores = np.linspace(0, 1, 1000)
    thr = reject_threshold_high(scores, 0.01)
    assert (scores > thr).mean() <= 0.011


def test_low_threshold_respects_the_false_reject_budget():
    """Confidence-style score: reject BELOW the threshold.

    Regression test. Reusing the high-side helper here once produced a 99%
    false-reject rate that no other metric caught -- OOD recall reads 100%
    when the model refuses everything.
    """
    scores = np.linspace(0, 1, 1000)
    thr = reject_threshold_low(scores, 0.01)
    assert (scores < thr).mean() <= 0.011


def test_the_two_thresholds_are_mirror_images():
    scores = np.linspace(0, 1, 1001)
    assert reject_threshold_low(scores, 0.05) < reject_threshold_high(scores, 0.05)
