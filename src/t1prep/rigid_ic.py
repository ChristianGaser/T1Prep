"""
Inverse-consistent rigid registration to an unbiased mid-space for N timepoints.

This module provides a practical, symmetric mid-space approach using SimpleITK:
- Iteratively registers all timepoints rigidly to a current template
- Resamples them into the template space and averages to update the template
- Returns (and can save) the final template and each timepoint's transform to it

CLI usage (writes template and transforms):
  python -m t1prep.rigid_ic \
    --inputs tp1.nii.gz tp2.nii.gz [tp3.nii.gz ...] \
    --out-dir subject/longitudinal/template \
    --iterations 2 --threads 8

Notes
-----
- This is a rigid-only solution using MI with a multi-resolution schedule.
- For brain-only alignment, provide precomputed brain-masked T1 images.
- Transforms are saved as ITK .tfm files (Euler3DTransform).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


def _import_sitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as e:  # pragma: no cover
        print(
            "ERROR: SimpleITK is required. Install with 'pip install SimpleITK' (>=2.1).\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        raise
    return sitk


@dataclass
class RigidICOutputs:
    template: Any
    transforms_to_template: List[Any]
    resampled_in_template: Optional[List[Any]] = None


def _rigid_register(
    sitk: Any,  # SimpleITK module
    moving: Any,
    fixed: Any,
    mask_moving: Optional[Any] = None,
    mask_fixed: Optional[Any] = None,
    metric: str = "mi",
) -> Any:
    """Register moving -> fixed with a rigid (Euler3D) transform.

    Parameters
    ----------
    moving, fixed : sitk.Image
        Input images (cast to Float32 by caller).
    mask_moving, mask_fixed : sitk.Image, optional
        Foreground masks (1 inside, 0 outside) to constrain metric, if provided.
    metric : {"mi","ncc","cc"}
        Similarity metric; MI is robust across scanners.
    """
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS
    )

    R = sitk.ImageRegistrationMethod()

    metric = metric.lower()
    if metric == "mi":
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.2)
    elif metric in ("ncc", "cc"):
        # Normalized Correlation
        R.SetMetricAsCorrelation()
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if mask_fixed is not None:
        R.SetMetricFixedMask(mask_fixed)
    if mask_moving is not None:
        R.SetMetricMovingMask(mask_moving)

    R.SetInterpolator(sitk.sitkLinear)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=4.0, minStep=1e-4, numberOfIterations=300, relaxationFactor=0.5
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(initial, inPlace=False)
    return R.Execute(fixed, moving)


def _resample(
    sitk: Any, moving: Any, reference: Any, tform: Any, interp: str = "linear"
) -> Any:
    interp_mode = sitk.sitkLinear if interp == "linear" else sitk.sitkNearestNeighbor
    return sitk.Resample(
        moving,
        reference,
        tform,
        interp_mode,
        0.0,
        sitk.sitkFloat32,
    )


def inverse_consistent_rigid_N(
    images: Sequence[Any],
    n_iter: int = 2,
    masks: Optional[Sequence[Optional[Any]]] = None,
    metric: str = "mi",
    return_resampled: bool = False,
) -> RigidICOutputs:
    """Inverse-consistent (unbiased mid-space) rigid registration for N timepoints.

    Parameters
    ----------
    images : sequence of sitk.Image
        Input images. They will be cast to Float32.
    n_iter : int
        Number of template refinement iterations (2–3 is typical).
    masks : sequence of sitk.Image or None
        Optional brain masks per image (same length as images) to restrict metric.
    metric : str
        Similarity metric: "mi" (default) or "ncc".
    return_resampled : bool
        Whether to return the per-iteration resampled images from the last iteration.

    Returns
    -------
    RigidICOutputs
        template, transforms_to_template (one per input), and optionally the
        resampled images from the last iteration.
    """
    sitk = _import_sitk()

    if len(images) < 2:
        raise ValueError("Need at least two timepoints for longitudinal rigid IC.")

    imgs = [sitk.Cast(im, sitk.sitkFloat32) for im in images]
    has_masks = masks is not None and len(masks) == len(imgs)

    # Initialize template as the first image
    template = imgs[0]

    transforms: List[Any] = [sitk.Euler3DTransform() for _ in imgs]
    for t in transforms:
        t.SetIdentity()

    resampled_last: Optional[List[Any]] = None

    for _ in range(max(1, int(n_iter))):
        resampled: List[Any] = []
        transforms = []
        for i, im in enumerate(imgs):
            m_mov = None
            m_fix = None
            if has_masks:
                m_mov = masks[i]
                m_fix = None  # mask only fixed or both; using fixed-only is common
            Ti = _rigid_register(sitk, moving=im, fixed=template, mask_moving=m_mov, mask_fixed=m_fix, metric=metric)
            transforms.append(Ti)
            resampled.append(_resample(sitk, im, template, Ti, interp="linear"))

        # Update template by averaging the resampled images
        acc = resampled[0]
        for j in range(1, len(resampled)):
            acc = acc + resampled[j]
        template = acc * (1.0 / float(len(resampled)))
        resampled_last = resampled

    return RigidICOutputs(template=template, transforms_to_template=transforms, resampled_in_template=resampled_last if return_resampled else None)


def save_outputs(
    outputs: RigidICOutputs,
    out_dir: str,
    input_paths: Optional[Sequence[str]] = None,
    template_name: str = "rigid_template_T1.nii.gz",
    transform_prefix: str = "tp",
) -> List[str]:
    """Save the template and transforms to disk.

    Returns list of transform file paths (one per timepoint).
    """
    sitk = _import_sitk()
    os.makedirs(out_dir, exist_ok=True)
    sitk.WriteImage(outputs.template, os.path.join(out_dir, template_name))

    tpaths: List[str] = []
    for i, T in enumerate(outputs.transforms_to_template):
        # Use index or derived name
        tag = f"{i}"
        if input_paths:
            base = os.path.basename(input_paths[i])
            tag = os.path.splitext(os.path.splitext(base)[0])[0]  # strip .nii.gz/.nii
        tpath = os.path.join(out_dir, f"{transform_prefix}{i}_toTemplate.tfm")
        sitk.WriteTransform(T, tpath)
        tpaths.append(tpath)
    return tpaths


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inverse-consistent rigid registration (mid-space) for N timepoints.")
    p.add_argument("--inputs", nargs="+", required=True, help="Input T1 images (NIfTI)")
    p.add_argument("--out-dir", required=True, help="Output directory for template and transforms")
    p.add_argument("--iterations", type=int, default=2, help="Template refinement iterations (default: 2)")
    p.add_argument("--metric", choices=["mi", "ncc", "cc"], default="mi", help="Similarity metric")
    p.add_argument("--threads", type=int, default=0, help="ITK threads (0=auto)")
    p.add_argument("--save-resampled", action="store_true", help="Also save last-iteration resampled images")
    return p.parse_args(argv)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    sitk = _import_sitk()
    args = _parse_args(argv)

    if args.threads and args.threads > 0:
        try:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
        except Exception:
            pass

    imgs = [sitk.ReadImage(p) for p in args.inputs]
    out = inverse_consistent_rigid_N(imgs, n_iter=args.iterations, metric=args.metric, return_resampled=args.save_resampled)

    tpaths = save_outputs(out, args.out_dir, input_paths=args.inputs)

    if args.save_resampled and out.resampled_in_template:
        os.makedirs(args.out_dir, exist_ok=True)
        for i, im in enumerate(out.resampled_in_template):
            sitk.WriteImage(im, os.path.join(args.out_dir, f"tp{i}_inTemplate.nii.gz"))

    print(f"Saved template and {len(tpaths)} transforms to {args.out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
