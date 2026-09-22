"""The arithmetic behind ``evaluation/tools/eval_phantom.py``.

The phantom evaluation itself is a manual, ~15-minute test.  What it reports
is only as good as these helpers, so they are checked here on synthetic
volumes whose answers are known in closed form.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "tools" / "eval_phantom.py"
_spec = importlib.util.spec_from_file_location("eval_phantom", _PATH)
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)


def test_fractions_split_the_label_and_sum_to_one():
    label = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    f = ep.tissue_fractions(label)
    total = f["CSF"] + f["GM"] + f["WM"]
    np.testing.assert_allclose(total[2:], 1.0)  # inside the brain
    np.testing.assert_allclose(f["GM"][5], 0.5)  # 2.5 = half GM, half WM
    np.testing.assert_allclose(f["WM"][5], 0.5)
    np.testing.assert_allclose(f["WM"][-2:], 1.0)  # WMH counts as WM
    np.testing.assert_allclose(f["CSF"][1], 0.5)  # half CSF, half background


def test_soft_dice_bounds():
    a = np.array([0.2, 0.8, 1.0])
    assert ep.soft_dice(a, a) == pytest.approx(1.0)
    assert ep.soft_dice(a, np.zeros_like(a)) == pytest.approx(0.0)


def _pv_label(radius, shape=(64, 64, 64)):
    """WM ball of the given radius in GM, with partial volume at its edge."""
    grid = np.indices(shape) - (np.array(shape)[:, None, None, None] - 1) / 2.0
    dist = np.sqrt((grid**2).sum(0))
    return (2.0 + np.clip(radius - dist + 0.5, 0, 1)).astype(np.float32)


@pytest.mark.parametrize("grow", [0.0, 0.3, -0.3, 1.0, -1.0])
def test_boundary_shift_measures_the_displacement(grow):
    # The GM/WM surface (2.5) of the prediction lies ``grow`` voxels of
    # 0.5 mm outside the true one.  The curvature term is grow / radius.
    pytest.importorskip("skimage")
    zooms = (0.5, 0.5, 0.5)
    shift = ep.boundary_shift(_pv_label(20.0), _pv_label(20.0 + grow), 2.5, zooms)
    assert shift == pytest.approx(grow * 0.5, abs=0.06 * abs(grow) * 0.5 + 1e-9)


def test_local_noise_recovers_the_sd():
    rng = np.random.default_rng(0)
    img = 100 + rng.normal(0, 5, (48, 48, 48)).astype(np.float32)
    mask = np.zeros(img.shape, bool)
    mask[4:-4, 4:-4, 4:-4] = True
    assert ep.local_noise(img, mask) == pytest.approx(5.0, rel=0.03)


def test_lowfreq_cv_sees_a_bias_field_but_not_noise():
    rng = np.random.default_rng(1)
    mask = np.zeros((48, 48, 48), bool)
    mask[4:-4, 4:-4, 4:-4] = True
    flat = 100 + rng.normal(0, 5, mask.shape).astype(np.float32)
    ramp = flat * np.linspace(0.8, 1.2, 48)[:, None, None]
    zooms = (1.0, 1.0, 1.0)
    assert ep.lowfreq_cv(flat, mask, zooms) < 0.01
    assert ep.lowfreq_cv(ramp, mask, zooms) > 0.05


@pytest.mark.parametrize(
    "direction, value, expected",
    [
        ("higher", 0.95, "ok"),
        ("higher", 0.90, "REGRESSED"),
        ("higher", 0.99, "improved"),
        ("lower", 0.99, "REGRESSED"),
        ("lower", 0.90, "improved"),
        ("both", 0.90, "REGRESSED"),
        ("both", 0.99, "REGRESSED"),
        ("both", float("nan"), "REGRESSED"),
    ],
)
def test_compare_follows_the_direction(direction, value, expected):
    pin = {"value": 0.95, "tol": 0.01, "direction": direction}
    assert ep.compare(value, pin) == expected


def test_strict_turns_improvements_into_failures():
    pin = {"value": 0.95, "tol": 0.01, "direction": "higher"}
    assert ep.compare(0.99, pin, strict=True) == "REGRESSED"


def test_pins_take_the_larger_of_spread_and_floor():
    runs = [
        {"metrics": {"GM_dice": 0.930, "thick_r": 0.90}},
        {"metrics": {"GM_dice": 0.934, "thick_r": 0.90}},
    ]
    pins = ep.make_pins(runs)
    assert pins["GM_dice"]["value"] == pytest.approx(0.932)
    assert pins["GM_dice"]["tol"] == pytest.approx(3 * 0.004)  # spread wins
    assert pins["thick_r"]["tol"] == pytest.approx(0.01)  # floor wins
    assert "WM_dice" not in pins  # never measured


def test_every_metric_has_a_valid_direction():
    for name, (direction, abs_floor, rel_floor) in ep.METRICS.items():
        assert direction in ("higher", "lower", "both"), name
        assert abs_floor > 0 or rel_floor > 0, f"{name} needs a tolerance floor"


def test_hook_records_the_partition_masks(tmp_path):
    # The manual run loads the hook as sitecustomize into T1Prep's processes.
    # Here it patches a real get_partition on the synthetic two-hemisphere
    # phantom of test_partition, and a hemisphere write then carries the
    # fill and clearance masks along.
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    (tmp_path / "hook").mkdir()
    (tmp_path / "hook" / "sitecustomize.py").write_text(ep.HOOK)
    out = tmp_path / "out"
    code = f"""
import numpy as np, nibabel as nib, torch, torch.nn.functional as F
from tests.test_partition import _phantom, _nifti, _guard_atlas
from t1prep._partition import get_partition
from t1prep.utils import resample_and_save_nifti, INTERP_KWARGS
lab, p0, reg = _phantom()
guard = _nifti(_guard_atlas(lab, reg), np.uint8)
lh, rh = get_partition(_nifti(p0), _nifti(lab), guard)
theta = torch.eye(3, 4)[None]
corners = INTERP_KWARGS["align_corners"]
grid = F.affine_grid(theta, (1, 1) + lh.shape, align_corners=corners)
img = _nifti(lh)
resample_and_save_nifti(img, grid, img.affine, img.header, "{tmp_path}/lh.seg.x.nii")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path / "hook"), str(root), str(root / "src")]
    )
    env["T1PREP_EVAL_GT_OUT"] = str(out)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert not (out / "gt_hook_error.json").exists()
    info = json.loads((out / "gt_hook.json").read_text())
    for side in ("lh", "rh"):
        assert info[side]["fill_voxels"] > 0
        assert info[side]["excluded_voxels"] > 0
        assert info[side]["fill_agreement"] == pytest.approx(1.0)
    assert (out / "lh.fill.x.nii").exists() and (out / "lh.excl.x.nii").exists()
    assert not (out / "rh.fill.x.nii").exists()  # only the side written


def test_csf_errors_land_in_their_zone():
    # A 1 mm "head": WM ball, GM shell, 4 mm CSF shell, and a CSF ventricle
    # deep inside.  The prediction loses the outer 2 mm of CSF only, as an
    # over-tight brain mask would: the error belongs to the rim alone.
    shape = (96, 96, 96)
    grid = np.indices(shape) - 47.5
    r = np.sqrt((grid**2).sum(0))
    label = np.select([r <= 20, r <= 36, r <= 40], [3.0, 2.0, 1.0], 0.0)
    vent = np.sqrt(((grid - np.array([0, 0, 8])[:, None, None, None]) ** 2).sum(0)) <= 6
    label[vent] = 1.0
    pred = np.where(r > 38, 0.0, label)
    f = ep.tissue_fractions
    zones = ep.csf_by_zone(label, f(label)["CSF"], f(pred)["CSF"], (1.0, 1.0, 1.0))
    lost = (label[(r > 38)] == 1.0).sum() / 1000.0
    assert zones["csf_err_rim_ml"] == pytest.approx(
        -lost, abs=0.01
    )  # rounded to 0.01 ml
    assert zones["csf_err_ventricles_ml"] == 0
    assert zones["csf_err_sulci_ml"] == 0


def test_wmh_scores_see_through_a_mask_that_flags_everything():
    # Flagging all of white matter "finds" every lesion; precision and the
    # flagged volume are what expose it.
    shape = (40, 40, 40)
    label = np.full(shape, 3.0, np.float32)
    label[10:13, 10:13, 10:13] = 4.0  # one 27-voxel lesion
    exact = np.where(label > 3.5, 0.9, 0.0).astype(np.float32)
    everything = np.full(shape, 0.2, np.float32)
    good = ep.wmh_scores(label, exact, 1.0)
    bad = ep.wmh_scores(label, everything, 1.0)
    assert good["found"] == bad["found"] == 1
    assert good["precision"] == pytest.approx(1.0)
    assert bad["precision"] < 0.01
    assert bad["flagged_ml"] == pytest.approx(64.0)
    assert good["dice_t050"] == pytest.approx(1.0) and bad["dice_t050"] == 0
