"""Small helpers shared by the segmentation stages."""

from typing import Union

import nibabel as nib
import numpy as np


def normalize_to_sum1(
    data1: Union[np.ndarray, nib.Nifti1Image],
    data2: Union[np.ndarray, nib.Nifti1Image],
    data3: Union[np.ndarray, nib.Nifti1Image],
):
    """
    Normalize three input arrays or Nifti1Images so that their sum is 1 at each voxel/data point.

    Each output is calculated as:
        paraX_norm = paraX / (data1 + data2 + data3)   (element-wise)

    Division by zero is handled by setting the sum to 1 at voxels/data points where it is zero.

    Parameters
    ----------
    data1 : np.ndarray or nibabel.Nifti1Image
        First input parameter (array or Nifti image).
    data2 : np.ndarray or nibabel.Nifti1Image
        Second input parameter (array or Nifti image).
    data3 : np.ndarray or nibabel.Nifti1Image
        Third input parameter (array or Nifti image).

    Returns
    -------
    norm1 : np.ndarray or nibabel.Nifti1Image
        Normalized first parameter. If input was Nifti1Image, output will be Nifti1Image.
    norm2 : np.ndarray or nibabel.Nifti1Image
        Normalized second parameter (same type as input).
    norm3 : np.ndarray or nibabel.Nifti1Image
        Normalized third parameter (same type as input).

    Notes
    -----
    - The type of the output for each parameter matches its input type:
        - If the input is a Nifti1Image, the output is a Nifti1Image (with the same affine and header).
        - If the input is an array, the output is an array.
    - Mixed types are supported (e.g., two arrays and one Nifti1Image).
    - For voxels/data points where data1 + data2 + data3 == 0, the denominator is set to 1 to avoid division by zero (output is then 0 for all three).
    - The input data are first clipped to a range of 0..1

    """

    def extract_data(x):
        if isinstance(x, nib.Nifti1Image):
            return x.get_fdata(), x
        else:
            return np.asarray(x), None

    data1, nifti1 = extract_data(data1)
    data2, nifti2 = extract_data(data2)
    data3, nifti3 = extract_data(data3)

    # Clip data first to a range of 0..1
    data1 = np.clip(data1, 0, 1)
    data2 = np.clip(data2, 0, 1)
    data3 = np.clip(data3, 0, 1)

    sum_data = data1 + data2 + data3
    sum_data[sum_data == 0] = 1  # Prevent division by zero

    norm1 = data1 / sum_data
    norm2 = data2 / sum_data
    norm3 = data3 / sum_data

    def wrap_nifti(norm, ref_nifti):
        if ref_nifti is not None:
            return nib.Nifti1Image(norm, ref_nifti.affine, ref_nifti.header)
        else:
            return norm

    return (
        wrap_nifti(norm1, nifti1),
        wrap_nifti(norm2, nifti2),
        wrap_nifti(norm3, nifti3),
    )
