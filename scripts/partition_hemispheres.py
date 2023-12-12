# python imports
import os
import sys
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# add main folder to python path and import ./ext/SynthSeg/predict_synthseg.py
home = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(home)
sys.path.append(os.path.join(home, 'ext'))

from ext.lab2im import utils as tools
from ext.lab2im import edit_volumes
from T1Prep import utils

def partition_hemispheres():
    """
    Partition the hemispheres of a brain based on an atlas and label segmentation.

    This script takes an atlas (segmentation) and a label segmentation as input,
    and partitions the hemispheres of the brain based on the provided label segmentation.
    The resulting hemisphere segmentations are saved as separate files.

    Usage:
        python partition_hemispheres.py --atlas <atlas_file> --label <label_file>

    Arguments:
        --atlas: Atlas (segmentation) from SynthSeg output.
        --label: Label segmentation.

    Returns:
        None
    """
    # parse arguments
    parser = ArgumentParser(description="partition_hemispheres", epilog='\n')

    # input/outputs
    parser.add_argument("--atlas",
                        help="Atlas (segmentation) from SynthSeg output")
    parser.add_argument("--label",
                        help="Label segmentation")

    # check for no arguments
    if len(sys.argv) < 2:
        print("\nMust provide at least --atlas or --label flags.")
        parser.print_help()
        sys.exit(1)

    # parse commandline
    args = vars(parser.parse_args())

    amap, aff_amap, h_amap = tools.load_volume(args['label'], im_only=False, dtype='float32')
    seg, aff_seg, h_seg = tools.load_volume(args['atlas'], im_only=False, dtype='float32')

    hemi_str = ['hemi-L', 'hemi-R']  # name for output file
    hemi_str2 = ['left', 'right']  # name for print

    for j in [0, 1]:
        print(f'Estimate surface label map for {hemi_str2[j]} hemisphere')
        hemi_name = args['label'].replace('.nii', f'_{hemi_str[j]}.nii')
        hemi = utils.amap2hemiseg(amap, aff_amap, seg, aff_seg, hemi=j + 1)

        # crop hemi image and add 5 voxels
        crop_idx = utils.bbox_volume(hemi > 1, pad=5)
        hemi, aff_hemi = edit_volumes.crop_volume_with_idx(hemi, crop_idx, aff=aff_amap, n_dims=3, return_copy=False)

        tools.save_volume(hemi, aff_hemi, h_amap, hemi_name, dtype='float32')

# Call the function to execute the script
partition_hemispheres()
