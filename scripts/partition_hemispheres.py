# python imports
import os
import sys
import numpy as np
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# add main folder to python path and import ./ext/SynthSeg/predict_synthseg.py
home = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(home)
sys.path.append(os.path.join(home, 'ext'))

from ext.lab2im import utils as tools
from ext.lab2im import edit_volumes
from T1Prep import utils

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
#amap, _, aff_amap, n_dims, n_channels, h_amap, res_amap = tools.get_volume_info(args['label'], True)
seg, aff_seg, h_seg = tools.load_volume(args['atlas'], im_only=False, dtype='float32')

hemi_str  = ['-L_seg', '-R_seg'] # name for output file
hemi_name = ['left', 'left']     # name for print

for j in [0, 1]:
    print('Estimate hemispheric amap label for %s hemisphere' % hemi_name[j])
    hemi_name = args['label'].replace('_seg.nii', '_%s_amap.nii' % hemi_str[j])
    hemi = utils.amap2hemiseg(amap, aff_amap, seg, aff_seg, hemi=j+1)

    # crop hemi image and add 5 voxels
    crop_idx = utils.bbox_volume(hemi > 1, pad=5)
    hemi, aff_hemi = edit_volumes.crop_volume_with_idx(hemi, crop_idx, aff=aff_amap, n_dims=3, return_copy=False)

    tools.save_volume(hemi, aff_hemi, h_amap, hemi_name, dtype='uint8')
            
