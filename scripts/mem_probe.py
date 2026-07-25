#!/usr/bin/env python
"""Per-stage memory probe for the deepmriprep pipeline as driven by T1Prep.

Wraps each ``Preprocess.run_*`` stage (and the segmentation sub-steps) with an
RSS sampler so we can see which stage drives the peak memory. Uses ``ps`` for
sampling (no extra dependencies) plus ``resource.getrusage`` as a monotonic
high-water cross-check.

Usage
-----
    ./env/bin/python scripts/mem_probe.py [T1.nii.gz] [--gpu] [--shape 256]

With no input path a brain-like ellipsoid phantom is generated at the given
cube size (default 256, 1 mm). The heavy stages allocate at the fixed template
grid, so phantom numbers track a real run's peak closely. Pass a real T1 for
exact figures. ``--gpu`` allows CUDA/MPS; default forces CPU (the 24 GB case).
"""

import argparse
import ctypes
import ctypes.util
import gc
import os
import resource
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"        # CPU Fallback for MPS
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" # More GPU memory

# ---------------------------------------------------------------------------
# RSS sampling
# ---------------------------------------------------------------------------
def _rss_bytes(pid):
    """Current resident set size in bytes via ``ps`` (KB on macOS/Linux)."""
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True
        ).strip()
        return int(out) * 1024
    except Exception:
        return 0


# Optional MPS memory reader, set in main() when running on MPS.
MPS_READER = None


class Sampler(threading.Thread):
    """Background thread tracking global and per-stage peak RSS (and MPS)."""

    def __init__(self, interval=0.04):
        super().__init__(daemon=True)
        self.interval = interval
        self.pid = os.getpid()
        self._stop = threading.Event()
        self.stage = "startup"
        self.global_peak = 0
        self.stage_peaks = {}
        self.mps_global_peak = 0
        self.mps_stage_peaks = {}

    def run(self):
        while not self._stop.is_set():
            rss = _rss_bytes(self.pid)
            if rss > self.global_peak:
                self.global_peak = rss
            if rss > self.stage_peaks.get(self.stage, 0):
                self.stage_peaks[self.stage] = rss
            if MPS_READER is not None:
                mps = MPS_READER()
                if mps > self.mps_global_peak:
                    self.mps_global_peak = mps
                if mps > self.mps_stage_peaks.get(self.stage, 0):
                    self.mps_stage_peaks[self.stage] = mps
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()


GB = 1024 ** 3
SAMPLER = Sampler()
ROWS = []


def _maxrss_bytes():
    """Peak RSS high-water mark. macOS reports bytes, Linux KB."""
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return val if sys.platform == "darwin" else val * 1024


def wrap(obj, name, label):
    """Wrap bound method ``obj.name`` to record memory around the call."""
    orig = getattr(obj, name)

    def wrapped(*args, **kwargs):
        prev_stage = SAMPLER.stage
        SAMPLER.stage = label
        SAMPLER.stage_peaks.setdefault(label, _rss_bytes(os.getpid()))
        rss_before = _rss_bytes(os.getpid())
        maxrss_before = _maxrss_bytes()
        t0 = time.time()
        try:
            return orig(*args, **kwargs)
        finally:
            dt = time.time() - t0
            rss_after = _rss_bytes(os.getpid())
            stage_peak = SAMPLER.stage_peaks.get(label, rss_after)
            maxrss_after = _maxrss_bytes()
            mps_peak = SAMPLER.mps_stage_peaks.get(label, 0)
            ROWS.append(
                {
                    "stage": label,
                    "dt": dt,
                    "rss_before": rss_before,
                    "rss_after": rss_after,
                    "stage_peak": stage_peak,
                    "maxrss_delta": maxrss_after - maxrss_before,
                    "maxrss_after": maxrss_after,
                    "mps_peak": mps_peak,
                }
            )
            # Stream the row immediately so an OOM-kill still leaves data.
            print(
                f"[row] {label:<20} {dt:6.1f}s  peakRSS={stage_peak / GB:5.2f}  "
                f"maxrss={maxrss_after / GB:5.2f}  mps={mps_peak / GB:5.2f} GB",
                flush=True,
            )
            SAMPLER.stage = prev_stage

    setattr(obj, name, wrapped)


# ---------------------------------------------------------------------------
# Free-between-stages driver
# ---------------------------------------------------------------------------
_LIBC = None


def _reclaim():
    """Collect Python garbage and hint the allocator to return freed pages."""
    gc.collect()
    global _LIBC
    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
        except Exception:
            _LIBC = False
    if _LIBC and hasattr(_LIBC, "malloc_trim"):
        try:
            _LIBC.malloc_trim(0)  # glibc only; no-op absent on macOS
        except Exception:
            pass


def run_with_freeing(prep, in_path):
    """Drive the pipeline manually, freeing each intermediate as soon as no
    later stage needs it (prototype of option 1)."""
    import nibabel as nib
    from deepmriprep.preprocess import IO

    t1 = nib.load(in_path)
    arr = t1.get_fdata()[..., 0] if len(t1.shape) == 4 else t1.get_fdata()
    outputs = {"t1": nib.Nifti1Image(arr, t1.affine, t1.header)}
    functions = {
        "bet": prep.run_bet,
        "affine": prep.run_affine_register,
        "segment_brain": prep.run_segment_brain,
        "segment_nogm": prep.run_segment_nogm,
        "warp": prep.run_warp_register,
        "smooth": prep.run_smooth,
        "atlas": prep.run_atlas_register,
    }
    steps = list(functions)
    for i, step in enumerate(steps):
        io = IO[step]
        imgs = tuple(outputs[k] for k in io["input"])
        kw = {"atlas_list": list(IO["atlas"]["output"])} if step == "atlas" else {}
        out = functions[step](*imgs, **kw)
        outputs.update(out)
        del imgs, out
        # Keep only keys any later stage still consumes.
        future = set()
        for later in steps[i + 1:]:
            future |= set(IO[later]["input"])
        for k in [k for k in outputs if k not in future]:
            del outputs[k]
        _reclaim()


# ---------------------------------------------------------------------------
# Isolated warp for clean peak measurement
# ---------------------------------------------------------------------------
def _synth_warp_inputs(prep):
    """Synthetic run_warp_register inputs at the real template grids.

    p0_large lives on the 0.5 mm affine template (339, 411, 339); the affine
    tissue maps live on the 1.5 mm warp template (113, 137, 113) = 1/3 of it.
    """
    import nibabel as nib
    import pandas as pd

    lshape = prep.affine_template.shape[:3]
    ashape = prep.warp_template.shape[:3]

    c = np.array(lshape) / 2.0
    zz, yy, xx = np.mgrid[0:lshape[0], 0:lshape[1], 0:lshape[2]].astype(np.float32)
    r = np.sqrt(((zz - c[0]) / (0.40 * lshape[0])) ** 2
                + ((yy - c[1]) / (0.42 * lshape[1])) ** 2
                + ((xx - c[2]) / (0.40 * lshape[2])) ** 2)
    del zz, yy, xx
    lab = np.zeros(lshape, np.float32)
    lab[r < 1.0] = 3.0
    lab[(r >= 0.80) & (r < 1.0)] = 2.0
    lab[(r >= 1.0) & (r < 1.10)] = 1.0
    del r
    p0_large = nib.Nifti1Image(lab, prep.affine_template.affine)

    c = np.array(ashape) / 2.0
    zz, yy, xx = np.mgrid[0:ashape[0], 0:ashape[1], 0:ashape[2]].astype(np.float32)
    r = np.sqrt(((zz - c[0]) / (0.40 * ashape[0])) ** 2
                + ((yy - c[1]) / (0.42 * ashape[1])) ** 2
                + ((xx - c[2]) / (0.40 * ashape[2])) ** 2)
    del zz, yy, xx
    gm = np.clip(1.0 - 8.0 * np.abs(r - 0.9), 0, 1).astype(np.float32)
    wm = np.clip(1.0 - 3.0 * r, 0, 1).astype(np.float32)
    del r
    aff = prep.warp_template.affine
    p1_affine = nib.Nifti1Image(gm, aff)
    p2_affine = nib.Nifti1Image(wm, aff)
    return p0_large, p1_affine, p2_affine, pd.Series([1.0])


def run_only_warp(prep, chunk):
    """Drive run_warp_register alone on synthetic inputs (isolated peak)."""
    SAMPLER.stage = "  warp.inputs"
    p0_large, p1_affine, p2_affine, wj_affine = _synth_warp_inputs(prep)
    _reclaim()
    rss0 = _rss_bytes(os.getpid())
    print(f"[warp] RSS with inputs built: {rss0 / GB:5.2f} GB", flush=True)

    # Sub-stage instrumentation: which part of the 113^3 registration is big?
    reg = prep.warp_register
    _orig_model, _orig_flows = reg.model, reg.syn.apply_flows

    def _model(*a, **k):
        SAMPLER.stage = "  warp.model"
        try:
            return _orig_model(*a, **k)
        finally:
            SAMPLER.stage = "  warp.register"

    def _flows(*a, **k):
        SAMPLER.stage = "  warp.apply_flows"
        try:
            return _orig_flows(*a, **k)
        finally:
            SAMPLER.stage = "  warp.register"

    reg.model, reg.syn.apply_flows = _model, _flows

    from t1prep._conv_chunk import chunked_conv3d

    SAMPLER.stage = "  warp.register"
    with chunked_conv3d(enabled=chunk) as mode:
        out = prep.run_warp_register(p0_large, p1_affine, p2_affine, wj_affine)
        split = mode.chunked_calls if mode else 0
    SAMPLER.stage = "  warp.done"
    print(f"[warp] convolutions split: {split}", flush=True)
    held = sum(
        int(np.prod(v.shape)) * 4 for v in out.values() if hasattr(v, "dataobj")
    )
    print(f"[warp] outputs held: {held / GB:5.2f} GB in {len(out)} entries", flush=True)
    return out, rss0


# ---------------------------------------------------------------------------
# Isolated segment_nogm (baseline vs chunked) for clean peak measurement
# ---------------------------------------------------------------------------
def _one_hot_select(p0, classes, n_cls=4):
    """Build only the requested one-hot channels directly, in order — avoids
    materializing the full 4-channel buffer plus a separate stacked copy."""
    import torch

    p0c = p0.clip(max=n_cls - 1)[:, 0]
    chans = []
    for c in classes:
        ch = torch.zeros_like(p0c)
        mask = p0c.gt(c - 1) & p0c.le(c + 1)
        ch[mask] = 1 - (p0c[mask] - c).abs()
        chans.append(ch)
    return torch.stack(chans, dim=1)


def _nogm_forward(nogm_seg, p0, chunk):
    """NoGMSegmentation.__call__ with optional memory-lean one-hot."""
    import torch
    import torch.nn.functional as F
    from deepmriprep.segment import one_hot

    SAMPLER.stage = "  nogm.run_model"
    nogm = nogm_seg.run_model(p0[:, :, 1:-2, 15:-12, :-3])
    nogm = F.pad(nogm, (0, 3, 15, 12, 1, 2)).to(nogm_seg.device)
    SAMPLER.stage = "  nogm.onehot"
    if chunk:
        p = _one_hot_select(p0, [2, 3, 1])  # 3 channels, no 4-ch buffer
    else:
        oh = one_hot(p0)
        p = torch.stack([oh[:, 2], oh[:, 3], oh[:, 1]], dim=1)
    p = nogm_seg.apply_nogm(p, nogm)
    return p, nogm


def run_only_nogm(prep, chunk, t1_shape=(256, 256, 256)):
    """Drive segment_nogm alone on synthetic template-res p0_large so the peak
    reflects this stage only (no carried memory from earlier stages)."""
    import numpy as np
    import nibabel as nib
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import spline_resize as sr
    from torchreg.utils import INTERP_KWARGS
    from deepmriprep.utils import nifti_to_tensor

    tshape = (336, 384, 336)
    c = np.array(tshape) / 2.0
    zz, yy, xx = np.mgrid[0:tshape[0], 0:tshape[1], 0:tshape[2]].astype(np.float32)
    r = np.sqrt(((zz - c[0]) / (0.40 * tshape[0])) ** 2
                + ((yy - c[1]) / (0.42 * tshape[1])) ** 2
                + ((xx - c[2]) / (0.40 * tshape[2])) ** 2)
    lab = np.zeros(tshape, np.float32)
    lab[r < 1.0] = 3.0
    lab[(r >= 0.80) & (r < 1.0)] = 2.0
    lab[(r >= 1.0) & (r < 1.10)] = 1.0
    p0_large = nib.Nifti1Image(lab, np.eye(4))
    t1 = nib.Nifti1Image(np.zeros(t1_shape, np.float32), np.eye(4))
    affine = pd.DataFrame(np.eye(4))

    SAMPLER.stage = "  nogm.model+onehot"
    p0t = nifti_to_tensor(p0_large)[None, None].to(prep.device)
    p_large, nogm = _nogm_forward(prep.nogm_segment, p0t, chunk)

    SAMPLER.stage = "  nogm.p_affine"
    p_affine = F.interpolate(p_large if chunk else p_large.clone(), scale_factor=1 / 3, mode="area")

    inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float().to(prep.device))
    grid = F.affine_grid(inv_affine[None, :3], [1, 3, *t1_shape],
                         align_corners=INTERP_KWARGS["align_corners"])

    SAMPLER.stage = "  nogm.gridsample"
    ac = INTERP_KWARGS["align_corners"]
    if chunk:
        p = torch.cat(
            [sr.grid_sample(p_large[:, i:i + 1], grid, align_corners=ac, mask_value=0)
             for i in range(p_large.shape[1])], dim=1)[0]
    else:
        p = sr.grid_sample(p_large, grid, align_corners=ac, mask_value=0)[0]
    p = p.clip(min=0, max=1).cpu()
    return p, p_large, nogm, p_affine


# ---------------------------------------------------------------------------
# Phantom
# ---------------------------------------------------------------------------
def make_phantom(path, n=256):
    """Write a 3-tissue ellipsoid phantom NIfTI at ``n`` mm-cube resolution."""
    import nibabel as nib

    c = n / 2.0
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    r = np.sqrt(
        ((xx - c) / (0.34 * n)) ** 2
        + ((yy - c) / (0.42 * n)) ** 2
        + ((zz - c) / (0.38 * n)) ** 2
    )
    vol = np.zeros((n, n, n), dtype=np.float32)
    vol[r < 1.0] = 110.0          # WM-ish core
    vol[(r >= 0.82) & (r < 1.0)] = 75.0   # GM-ish rim
    vol[(r >= 1.0) & (r < 1.08)] = 35.0   # CSF-ish shell
    rng = np.random.default_rng(0)
    vol += rng.normal(0, 3, vol.shape).astype(np.float32) * (vol > 0)
    vol = np.clip(vol, 0, None)
    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(vol, affine), path)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(b):
    return f"{b / GB:6.2f}"


def report():
    has_mps = SAMPLER.mps_global_peak > 0
    print("\n" + "=" * 100)
    print("PER-STAGE MEMORY (GB)" + ("   [mps peak = torch.mps.driver_allocated]" if has_mps else ""))
    print("=" * 100)
    print(
        f"{'stage':<22}{'secs':>7}{'RSS in':>9}{'RSS out':>9}"
        f"{'stage peak':>12}{'maxrss':>9}{'mps peak':>10}"
    )
    print("-" * 100)
    for r in ROWS:
        print(
            f"{r['stage']:<22}{r['dt']:>7.1f}{_fmt(r['rss_before']):>9}"
            f"{_fmt(r['rss_after']):>9}{_fmt(r['stage_peak']):>12}"
            f"{_fmt(r['maxrss_after']):>9}{_fmt(r.get('mps_peak', 0)):>10}"
        )
    print("-" * 100)
    print(
        f"{'GLOBAL PEAK':<22}{'':>7}{'':>9}{'':>9}"
        f"{_fmt(SAMPLER.global_peak):>12}{'':>9}{_fmt(SAMPLER.mps_global_peak):>10}"
    )
    if ROWS:
        key = "mps_peak" if has_mps else "stage_peak"
        top = max(ROWS, key=lambda r: r.get(key, 0))
        metric = "MPS" if has_mps else "RSS"
        print(f"\nPeak-driving stage ({metric}): {top['stage']}  ({_fmt(top.get(key, 0)).strip()} GB)")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default=None, help="T1 NIfTI; phantom if omitted")
    ap.add_argument("--gpu", action="store_true", help="allow CUDA/MPS (default: force CPU)")
    ap.add_argument("--shape", type=int, default=256, help="phantom cube size (mm)")
    ap.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/Dropbox/GitHub/deepmriprep/deepmriprep/data/models"),
        help="model dir with hydrated weights (installed copies may be Dropbox placeholders)",
    )
    ap.add_argument(
        "--free",
        action="store_true",
        help="prototype option 1: free intermediates + gc between stages",
    )
    ap.add_argument(
        "--autocast",
        choices=["none", "fp16", "bf16"],
        default="none",
        help="wrap the whole pipeline in autocast of this dtype "
        "(MPS lacks fp16/bf16 ConvTranspose3d; CPU supports both)",
    )
    ap.add_argument(
        "--only-nogm",
        action="store_true",
        help="run segment_nogm alone on synthetic p0_large (isolated peak)",
    )
    ap.add_argument(
        "--chunk-nogm",
        action="store_true",
        help="use the memory-lean chunked segment_nogm (per-channel grid_sample, "
        "3-channel one-hot, no clone)",
    )
    ap.add_argument(
        "--only-warp",
        action="store_true",
        help="run run_warp_register alone on synthetic inputs (isolated peak)",
    )
    ap.add_argument(
        "--chunk-conv",
        action="store_true",
        help="split oversized CPU conv3d into slabs (t1prep._conv_chunk), which "
        "is what the pipeline does for the warp stage",
    )
    ap.add_argument(
        "--verify-warp",
        action="store_true",
        help="run the warp stage with and without slab-wise conv and report the "
        "max abs difference per output",
    )
    args = ap.parse_args()

    if not args.gpu:
        # Force CPU to reproduce the high-memory CPU scenario.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Point T1Prep/deepmriprep at hydrated weights before importing them.
    if args.model_dir and os.path.isdir(args.model_dir):
        os.environ["T1PREP_MODEL_DIR"] = args.model_dir

    import torch

    if not args.gpu:
        # CUDA_VISIBLE_DEVICES doesn't disable MPS; force it off so every stage
        # (incl. CustomBrainSegmentation.inference_device) stays on CPU.
        torch.backends.mps.is_available = lambda: False

    from t1prep._models import _redirect_deepmriprep_paths, MODEL_DIR

    _redirect_deepmriprep_paths()
    # T1Prep's redirect misses submodules that copied DATA_PATH at import
    # (e.g. deepmriprep.register). Patch every one that has it.
    fake_data = str(MODEL_DIR.parent)
    import importlib

    for mod in ("utils", "preprocess", "segment", "register", "atlas"):
        try:
            m = importlib.import_module(f"deepmriprep.{mod}")
            if hasattr(m, "DATA_PATH"):
                m.DATA_PATH = fake_data
        except Exception:
            pass
    print(f"Model dir: {MODEL_DIR}")

    no_gpu = not args.gpu
    is_mps = False
    if args.gpu:
        is_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()
        no_gpu = not (torch.cuda.is_available() or torch.backends.mps.is_available())

    # Enable accurate MPS memory tracking.
    if is_mps and hasattr(torch.mps, "driver_allocated_memory"):
        global MPS_READER
        MPS_READER = torch.mps.driver_allocated_memory

    from t1prep.segment import CustomPreprocess, CustomBrainSegmentation

    scratch = os.environ.get("TMPDIR", "/tmp")
    in_path = args.input
    isolated = args.only_nogm or args.only_warp or args.verify_warp
    if in_path is None and not isolated:
        in_path = os.path.join(scratch, "mem_probe_phantom.nii.gz")
        print(f"Generating phantom ({args.shape}^3) -> {in_path}")
        make_phantom(in_path, args.shape)

    dev = "mps" if is_mps else ("cuda" if not no_gpu else "cpu (forced)")
    print(f"Device: {dev}   autocast: {args.autocast}")
    SAMPLER.start()

    prep = CustomPreprocess(no_gpu=no_gpu)

    if args.verify_warp:
        # Cross-check the slab-wise convolution on the real warp shapes; the
        # unit tests only cover toy volumes.
        from t1prep._conv_chunk import chunked_conv3d

        print("Mode: verify warp (stock vs chunked conv)", flush=True)
        inputs = _synth_warp_inputs(prep)
        with chunked_conv3d():
            chunked = prep.run_warp_register(*inputs)
        base = prep.run_warp_register(*inputs)
        SAMPLER.stop()
        worst = 0.0
        for key in sorted(base):
            b, c = base[key], chunked[key]
            if hasattr(b, "dataobj"):
                a1 = np.asanyarray(b.dataobj, dtype=np.float64)
                a2 = np.asanyarray(c.dataobj, dtype=np.float64)
                d = float(np.abs(a1 - a2).max())
            else:
                d = float(abs(np.asarray(b) - np.asarray(c)).max())
            worst = max(worst, d)
            print(f"  {key:<10} max|diff| = {d:.3e}")
        print(f"\nworst deviation across all outputs: {worst:.3e}"
              f"  -> {'BITWISE IDENTICAL' if worst == 0 else 'DIFFERS'}")
        return

    if args.only_warp:
        tag = "CHUNKED" if args.chunk_conv else "BASELINE"
        print(f"Mode: isolated warp [{tag}]", flush=True)
        import time as _t
        t0 = _t.time()
        rss0 = 0
        try:
            _out, rss0 = run_only_warp(prep, args.chunk_conv)
        except Exception:
            print("\n[warp failed]\n" + traceback.format_exc())
        finally:
            SAMPLER.stop()
            time.sleep(0.1)
            print(f"\nisolated warp [{tag}]  ({_t.time() - t0:.1f}s)")
            for st, pk in SAMPLER.stage_peaks.items():
                print(f"  {st:<24} peakRSS={pk / GB:5.2f} GB")
            print(f"  {'GLOBAL PEAK RSS':<24} peakRSS={SAMPLER.global_peak / GB:5.2f} GB")
            if rss0:
                print(f"  {'STAGE COST (peak-inputs)':<24} "
                      f"{(SAMPLER.global_peak - rss0) / GB:5.2f} GB")
        return

    if args.only_nogm:
        tag = "CHUNKED" if args.chunk_nogm else "BASELINE"
        print(f"Mode: isolated segment_nogm [{tag}]", flush=True)
        import time as _t
        t0 = _t.time()
        try:
            _ = run_only_nogm(prep, args.chunk_nogm)
        except Exception:
            print("\n[nogm failed]\n" + traceback.format_exc())
        finally:
            SAMPLER.stop()
            time.sleep(0.1)
            print(f"\nisolated segment_nogm [{tag}]  ({_t.time() - t0:.1f}s)")
            for st, pk in SAMPLER.stage_peaks.items():
                print(f"  {st:<24} peakRSS={pk / GB:5.2f} GB")
            print(f"  {'GLOBAL PEAK RSS':<24} peakRSS={SAMPLER.global_peak / GB:5.2f} GB")
        return

    prep.brain_segment = CustomBrainSegmentation(no_gpu=no_gpu)

    # Apply autocast ONLY around the neural-net forwards (where the peak
    # activations live), casting outputs back to fp32 so resampling / numpy
    # conversion downstream keep working.
    if args.autocast != "none":
        dt = torch.float16 if args.autocast == "fp16" else torch.bfloat16
        dev_type = "mps" if is_mps else "cpu"

        def _to_float(x):
            if torch.is_tensor(x):
                return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
            if isinstance(x, (list, tuple)):
                return type(x)(_to_float(v) for v in x)
            if isinstance(x, dict):
                return {k: _to_float(v) for k, v in x.items()}
            return x

        class _AC:
            def __init__(self, m):
                self.m = m

            def __call__(self, *a, **k):
                with torch.autocast(device_type=dev_type, dtype=dt):
                    out = self.m(*a, **k)
                return _to_float(out)

            def __getattr__(self, name):
                return getattr(self.m, name)

        s = prep.brain_segment
        s.model = _AC(s.model)
        s.patch_models = [_AC(m) for m in s.patch_models]
        prep.nogm_segment.model = _AC(prep.nogm_segment.model)
        prep.warp_register.model = _AC(prep.warp_register.model)

    # Wrap each pipeline stage.
    for m, label in [
        ("run_bet", "bet"),
        ("run_affine_register", "affine"),
        ("run_segment_brain", "segment_brain"),
        ("run_segment_nogm", "segment_nogm"),
        ("run_warp_register", "warp"),
        ("run_atlas_register", "atlas"),
    ]:
        if hasattr(prep, m):
            wrap(prep, m, label)
    # Sub-steps inside brain segmentation.
    for m, label in [("run_model", "  seg.run_model"), ("run_patch_models", "  seg.patches")]:
        if hasattr(prep.brain_segment, m):
            wrap(prep.brain_segment, m, label)

    from contextlib import nullcontext

    mode = "FREE between stages" if args.free else (
        f"{args.autocast} autocast (models only)" if args.autocast != "none" else "baseline")
    print(f"Mode: {mode}")
    try:
        if args.free:
            run_with_freeing(prep, in_path)
        else:
            prep.run(in_path, {}, run_all=True, seed=0, skip_unprocessed=False)
    except Exception:
        print("\n[pipeline stopped early]\n" + traceback.format_exc())
    finally:
        SAMPLER.stop()
        time.sleep(0.1)
        report()


if __name__ == "__main__":
    main()
