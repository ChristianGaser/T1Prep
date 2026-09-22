"""The calibrated WMH probability and the lesion selection in ``_lesions``.

``p4`` used to be the lesion *signal* (three times the intensity deficit),
not a probability.  These tests pin what the replacement has to do whatever
its fitted constants are: it is a probability, the local level does not eat
large lesions, and a lesion is kept or dropped as a whole.
"""

import numpy as np
import pytest

from t1prep._lesions import (
    WMH_CALIBRATION,
    WMH_PRIOR_EPS,
    _highpass,
    lesion_probability,
    select_lesions,
)


def test_probability_is_bounded_and_zero_outside_the_candidates():
    rng = np.random.default_rng(0)
    diff = rng.normal(0, 0.5, (20, 20, 20)).astype(np.float32)
    prior = rng.random((20, 20, 20)).astype(np.float32)
    cand = np.zeros(diff.shape, bool)
    cand[5:15, 5:15, 5:15] = True
    p = lesion_probability(diff, prior, cand)
    assert p.dtype == np.float32
    assert p.min() >= 0 and p.max() <= 1
    assert not p[~cand].any()


def test_probability_follows_the_logistic_model():
    a, b, c = WMH_CALIBRATION
    diff = np.array([0.0, 0.1, 0.3], np.float32)
    prior = np.array([0.5, 0.5, 0.5], np.float32)
    p = lesion_probability(diff, prior, np.ones(3, bool))
    logit = a + b * diff + c * np.log(prior + WMH_PRIOR_EPS)
    np.testing.assert_allclose(p, 1 / (1 + np.exp(-logit)), rtol=1e-5)


def test_probability_rises_with_the_deficit_and_the_prior():
    diff = np.linspace(-0.2, 0.6, 50, dtype=np.float32)
    cand = np.ones(50, bool)
    p_low = lesion_probability(diff, np.full(50, 0.05, np.float32), cand)
    p_high = lesion_probability(diff, np.full(50, 0.9, np.float32), cand)
    assert np.all(np.diff(p_low) >= 0)
    assert np.all(p_high >= p_low)


def _blob(shape, centre, radius, value):
    grid = np.indices(shape) - np.array(centre)[:, None, None, None]
    return np.where(np.sqrt((grid**2).sum(0)) <= radius, value, 0.0)


def test_a_lesion_is_kept_whole_with_its_probabilities():
    shape = (40, 40, 40)
    core = _blob(shape, (20, 20, 20), 3, 0.9)
    rim = _blob(shape, (20, 20, 20), 4, 0.3)
    prob = np.maximum(core, rim).astype(np.float32)
    kept, mask = select_lesions(prob, vx_vol=1.0, threshold=0.1, min_lesion_mm3=5.0)
    np.testing.assert_array_equal(kept, prob)
    assert mask.sum() == (prob > 0.1).sum()


def test_small_and_faint_blobs_are_dropped():
    shape = (40, 40, 40)
    faint = _blob(shape, (10, 10, 10), 5, 0.05)  # never above the threshold
    tiny = _blob(shape, (30, 30, 30), 0.5, 0.9)  # a single voxel of 1 mm3
    kept, mask = select_lesions(
        (faint + tiny).astype(np.float32), vx_vol=1.0, threshold=0.1, min_lesion_mm3=5.0
    )
    assert not kept.any() and not mask.any()


def test_the_size_limit_is_a_volume():
    # 19 voxels: dropped at 1 mm3 each under a 30 mm3 limit, kept at
    # 8 mm3 each (2 mm voxels) -- the limit must not depend on the grid.
    prob = _blob((30, 30, 30), (15, 15, 15), 1.5, 0.95).astype(np.float32)
    small, _ = select_lesions(prob, vx_vol=1.0, threshold=0.1, min_lesion_mm3=30.0)
    large, _ = select_lesions(prob, vx_vol=8.0, threshold=0.1, min_lesion_mm3=30.0)
    assert not small.any()
    assert large.sum() == pytest.approx(prob.sum())


def test_diagonal_neighbours_form_one_lesion():
    # Two voxels touching only at a corner: 26-connectivity makes them one
    # lesion of 2 mm3, where either alone would fall under a 1.5 mm3 limit.
    prob = np.zeros((5, 5, 5), np.float32)
    prob[1, 1, 1] = prob[2, 2, 2] = 0.9
    kept, _ = select_lesions(prob, vx_vol=1.0, threshold=0.1, min_lesion_mm3=1.5)
    assert kept.sum() == pytest.approx(1.8)


def test_highpass_removes_a_smooth_offset():
    # What the bias correction left is a slow drift of the signal; it must
    # not read as a lesion.
    shape = (48, 48, 48)
    cand = np.zeros(shape, bool)
    cand[6:-6, 6:-6, 6:-6] = True
    drift = np.linspace(-0.05, 0.05, 48, dtype=np.float32)[:, None, None]
    diff = np.broadcast_to(drift, shape).copy()
    hp = _highpass(diff, cand, (1.0, 1.0, 1.0), sigma_mm=5.0)
    assert np.abs(hp[cand]).max() < 0.01


def test_a_large_lesion_keeps_its_deficit_only_with_a_robust_reference():
    # A lesion wider than the level's reach.  With every voxel in the
    # reference it raises the level and subtracts part of itself; with the
    # reference limited to low signal it keeps (almost) all of its 0.3.
    shape = (64, 64, 64)
    cand = np.zeros(shape, bool)
    cand[4:-4, 4:-4, 4:-4] = True
    grid = np.indices(shape) - 31.5
    lesion = np.sqrt((grid**2).sum(0)) <= 12
    diff = np.where(lesion, 0.3, 0.0).astype(np.float32)
    zooms = (1.0, 1.0, 1.0)
    robust = _highpass(diff, cand, zooms, sigma_mm=5.0, ref_max=0.1)[32, 32, 32]
    clipped = _highpass(diff, cand, zooms, sigma_mm=5.0, ref_max=None)[32, 32, 32]
    assert robust > 0.9 * 0.3
    assert clipped < 0.9 * 0.3
