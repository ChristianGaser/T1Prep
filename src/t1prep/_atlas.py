"""Template and atlas access on the working grid.

Locating a template file, resampling it onto a target grid, and turning a
label atlas into region masks.  Shared by the segmentation, the vessel
correction and the non-cortical GM rule.
"""

import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from deepmriprep.atlas import AtlasRegistration, shape_from_to
from deepmriprep.utils import nifti_to_tensor

from .utils import TEMPLATE_PATH_T1PREP


def resolve_template_file(name: str, ext: str) -> str:
    """Return the full path for *name* + *ext* inside TEMPLATE_PATH_T1PREP.

    Performs an exact lookup first and falls back to a case-insensitive scan
    of the directory so that callers using different capitalisation (e.g.
    ``"ibsr"`` vs the on-disk ``"IBSR"``) still resolve correctly.

    Parameters
    ----------
    name:
        Base name without extension (e.g. ``"IBSR"`` or ``"neuromorphometrics"``).
    ext:
        File extension including the leading dot (e.g. ``".csv"`` or
        ``".nii.gz"``).

    Returns
    -------
    str
        Absolute path to the resolved file.

    Raises
    ------
    FileNotFoundError
        If no matching file is found in the template directory.
    """
    # Fast exact path
    candidate = os.path.join(TEMPLATE_PATH_T1PREP, f"{name}{ext}")
    if os.path.exists(candidate):
        return candidate

    # Case-insensitive fallback
    target = f"{name.lower()}{ext.lower()}"
    for fname in os.listdir(TEMPLATE_PATH_T1PREP):
        if fname.lower() == target:
            return os.path.join(TEMPLATE_PATH_T1PREP, fname)

    raise FileNotFoundError(
        f"Template file '{name}{ext}' not found in {TEMPLATE_PATH_T1PREP}"
    )


def resample_to(img, target_affine, target_shape, device="cpu", channel=None,
                 nearest=False):
    """Trilinear resample *img* onto the grid given by affine and shape.

    ``get_atlas`` resizes the template array onto the target dimensions and so
    silently assumes both cover the same field of view.  That holds for the
    label atlases it is used with, but ``cat_bloodvessels.nii.gz`` lives on the
    SPM TPM grid (origin -90/-126/-72) while T1Prep works on the shooting
    template grid (origin -84/-120/-72).  A plain resize would misplace the
    prior by several millimetres, which matters most exactly where it is used
    -- around the insula.  So the real affines are honoured here.

    The sampling grid is built at roughly the template resolution and only
    then resized to the target shape, so the explicit coordinate array stays
    small even when the target is a 0.5 mm volume.

    Set ``nearest`` for label atlases.  Interpolating label *ids* linearly
    invents labels that lie between two unrelated regions, which silently
    turns a protection mask into nonsense.
    """
    img = nib.as_closest_canonical(img)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if channel is not None:
        data = data[..., channel]

    target_affine = np.asarray(target_affine, dtype=float)
    target_shape = np.asarray(target_shape, dtype=int)[:3]
    tgt_zoom = np.sqrt((target_affine[:3, :3] ** 2).sum(axis=0))
    src_zoom = np.asarray(img.header.get_zooms()[:3], dtype=float)

    # Intermediate grid: same field of view, roughly the template resolution.
    n_int = np.maximum(2, np.round(target_shape * tgt_zoom / src_zoom)).astype(int)
    inter = target_affine.copy()
    inter[:3, :3] = target_affine[:3, :3] * (target_shape / n_int)
    # Keep the outer field of view identical by shifting the first voxel centre.
    inter[:3, 3] = target_affine[:3, 3] + 0.5 * (
        inter[:3, :3] - target_affine[:3, :3]
    ) @ np.ones(3)

    # Intermediate voxel -> source voxel.
    to_src = np.linalg.inv(img.affine) @ inter
    grids = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in n_int], indexing="ij")
    coords = (
        to_src[:3, :3].astype(np.float32) @ np.stack([g.ravel() for g in grids])
        + to_src[:3, 3, None].astype(np.float32)
    )

    # grid_sample expects normalised coordinates in reversed axis order.
    shape_src = np.asarray(data.shape[:3], dtype=np.float32)
    norm = 2.0 * coords / np.maximum(shape_src - 1.0, 1.0)[:, None] - 1.0
    grid = torch.as_tensor(
        norm[::-1].T.reshape(*n_int, 3).copy(), device=device
    )[None]

    src = torch.as_tensor(
        np.ascontiguousarray(data, dtype=np.float32), device=device
    )[None, None]
    out = F.grid_sample(
        src,
        grid,
        mode="nearest" if nearest else "bilinear",
        align_corners=True,
        padding_mode="border",
    )
    mode = "nearest" if nearest else "trilinear"
    kwargs = {} if nearest else {"align_corners": False}
    out = F.interpolate(
        out, size=tuple(int(v) for v in target_shape), mode=mode, **kwargs
    )
    return out[0, 0].cpu().numpy()


def get_atlas(
    t1,
    affine,
    target_header,
    target_affine,
    atlas_name,
    warp_yx=None,
    device="cpu",
    is_label_atlas: bool = True,
):
    """Generate an atlas-aligned image in the target space.

    Parameters
    ----------
    t1 : nib.Nifti1Image
        Reference image in target space. Only the shape is used here
        when applying the deformation field.
    affine : np.ndarray
        Affine of the target image used for atlas registration.
    target_header : nib.Nifti1Header
        Header of the target image; copied to the returned atlas image.
    target_affine : np.ndarray
        Affine of the target image; used as transform for the returned atlas.
    atlas_name : str
        Base file name of the atlas (``<atlas_name>.nii.gz`` located in
        ``TEMPLATE_PATH_T1PREP``).
    warp_yx : nib.Nifti1Image, optional
        Optional deformation field from atlas space to target space. If
        provided, the atlas is first warped using this field before
        resampling to the requested output grid.
    device : str or torch.device, optional
        Device on which interpolation is performed (default: ``"cpu"``).
    is_label_atlas : bool, optional
        If ``True`` (default), the atlas is assumed to contain discrete
        labels. Nearest-neighbour interpolation is used and the result is
        stored as an integer type (``uint8`` if the maximum label is
        smaller than 256, otherwise ``int16``).

        If ``False``, the atlas is assumed to contain continuous values
        (e.g., tissue probability maps). Linear interpolation is used and
        the output is stored as floating point (``float32``).

    Returns
    -------
    nib.Nifti1Image
        Atlas image resampled into the target space.

    """
    header = target_header
    dim_hdr = target_header["dim"][1:4]
    dim = tuple(int(x) for x in dim_hdr)
    transform = target_affine

    atlas = nib.as_closest_canonical(
        nib.load(resolve_template_file(atlas_name, ".nii.gz"))
    )
    atlas_register = AtlasRegistration()

    if warp_yx is not None:
        warp_yx = nib.as_closest_canonical(warp_yx)
        yx = nifti_to_tensor(warp_yx)[None].to(device)
        shape = tuple(shape_from_to(atlas, warp_yx))
        scaled_yx = F.interpolate(
            yx.permute(0, 4, 1, 2, 3), shape, mode="trilinear", align_corners=False
        )
        warps = {shape: scaled_yx.permute(0, 2, 3, 4, 1)}
        # AtlasRegistration internally pins its tensors to its own (CPU)
        # device.  When ``device`` here is MPS/CUDA, the warp would be on a
        # different device than the atlas tensor, triggering grid_sample's
        # "input and grid to be on same device" error.  Align them.
        atlas = atlas_register(
            affine, warps[shape].to(atlas_register.device), atlas, t1.shape
        )

    # Resizing onto the target dimensions silently assumes both grids cover
    # the same field of view.  That holds after the warp above -- deepmriprep
    # returns the atlas on WARP_TEMPLATE, which shares the working field of
    # view -- and for templates already on the working grid such as cat_wmh.
    # It does not hold for the 1 mm atlases (IBSR, Neuromorphometrics), whose
    # field of view is 161x197x161 mm against the working 169.5x205.5x169.5:
    # a plain resize stretches them by 5% and displaces structures by up to
    # 4.2 mm at the edges.  So the affines decide, and the fast path is taken
    # only when they genuinely agree.
    src_fov = np.asarray(atlas.shape[:3]) * np.sqrt(
        (np.asarray(atlas.affine)[:3, :3] ** 2).sum(axis=0)
    )
    tgt_affine = np.asarray(transform, dtype=float)
    tgt_fov = np.asarray(dim) * np.sqrt((tgt_affine[:3, :3] ** 2).sum(axis=0))
    src_origin = np.asarray(atlas.affine)[:3, 3]
    aligned = np.allclose(src_fov, tgt_fov, atol=1e-3) and np.allclose(
        src_origin, tgt_affine[:3, 3], atol=1e-3
    )

    if aligned:
        atlas_tensor = nifti_to_tensor(atlas)[None, None].to(device)
        if is_label_atlas:
            atlas_np = F.interpolate(atlas_tensor, dim, mode="nearest")[0, 0]
        else:
            atlas_np = F.interpolate(
                atlas_tensor, dim, mode="trilinear", align_corners=False
            )[0, 0]
        atlas_np = atlas_np.cpu().numpy()
    else:
        atlas_np = resample_to(
            atlas, tgt_affine, dim, device=device, nearest=is_label_atlas
        )

    if is_label_atlas:
        atlas_np = np.round(atlas_np)
        atlas_np = atlas_np.astype(
            np.uint8 if atlas_np.max() < 256 else np.int16
        )
    else:
        atlas_np = atlas_np.astype(np.float32)

    return nib.Nifti1Image(atlas_np, transform, header)


def get_regions_mask(
    atlas: nib.Nifti1Image,
    atlas_name: str,
    region_name: list[str],
) -> np.ndarray:
    """Return a binary mask for a set of regions in a label atlas.

    This helper reads the ROI definition CSV associated with ``atlas_name``
    (``<atlas_name>.csv`` in ``TEMPLATE_PATH_T1PREP``), maps region
    name to their numeric IDs, and returns a boolean mask where
    voxels belonging to any of the requested regions are ``True``.

    Parameters
    ----------
    atlas : nib.Nifti1Image
        Label atlas image in the same space as the desired mask.
    atlas_name : str
        Base name of the atlas (e.g. ``"ibsr"``). The corresponding CSV
        file is expected at ``TEMPLATE_PATH_T1PREP/<atlas_name>.csv`` and
        must contain at least the columns ``ROIid`` and ``ROIabbr``.
    region_name : list of str
        List of ROI name (``"ROIname"``) to include in the mask (e.g.
        ``["Left Cerebellum White Matter", "Right Cerebellum White Matter"]``).

    Returns
    -------
    np.ndarray
        Boolean array with the same shape as ``atlas.get_fdata()``, where
        ``True`` indicates voxels belonging to any of the requested
        regions.

    """
    rois = pd.read_csv(resolve_template_file(atlas_name, ".csv"), sep=";")[
        ["ROIid", "ROIname"]
    ]
    regions = dict(zip(rois.ROIname, rois.ROIid))
    atlas_data = np.round(atlas.get_fdata())
    region_ids = [regions[r] for r in region_name if r in regions]
    return np.isin(atlas_data, region_ids)
