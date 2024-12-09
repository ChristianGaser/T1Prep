import os
import os.path
import subprocess as sub
import tempfile
import nibabel as nib


def applyMINCSmooth(inimg, mask=None, lmb=0.01,dist=75,subsamp=2,
                    extrapolate=False):

    # Python 3: tmpdir = tempfile.TemporaryDirectory()
    tmpdir = tempfile.mkdtemp()
    tmpfile=os.path.join(tmpdir,'tmpfile')
    tmpfilenii1="{}1.nii".format(tmpfile)
    tmpfilenii2="{}2.nii".format(tmpfile)
    tmpfilenii3="{}3.nii".format(tmpfile)
    tmpfilemnc1="{}1.mnc".format(tmpfile)
    tmpfilemnc2="{}2.mnc".format(tmpfile)

    tmpfileniimask="{}mask.nii".format(tmpfile)
    tmpfilemncmask="{}mask.mnc".format(tmpfile)

    haveMask = mask is not None

    nib.save(inimg,tmpfilenii1)
    convcmd = ['nii2mnc',tmpfilenii1,tmpfilemnc1]
    print(convcmd)
    sub.call(convcmd)
    # Can't do this.
    #tmpmncout = nib.Minc1Image(inimgdata, inimg.affine, inimg.header)
    #nib.save(tmpmncout,tmpfile)

    if haveMask:
        maskimg = nib.Nifti1Image(mask, inimg.affine, inimg.header)
        nib.save(maskimg,tmpfileniimask)
        convcmd = ['nii2mnc',tmpfileniimask,tmpfilemncmask]
        print(convcmd)
        sub.call(convcmd)
        

    splcmd = ['spline_smooth']
    if haveMask:
        splcmd = splcmd + ['-mask',tmpfilemncmask]
    if extrapolate:
        splcmd = splcmd + ['-extrapolate']
    splcmd = splcmd + ['-lambda',lmb,
                       '-distance',dist,
                       '-subsample',subsamp,
                       tmpfilemnc1,tmpfilemnc2]
    splcmd = ["{}".format(x) for x in splcmd]
    print(splcmd)
    sub.call(splcmd)
    convcmd = ['mnc2nii',tmpfilemnc2,tmpfilenii2]
    print(convcmd)
    sub.call(convcmd)
    voffcmd = ['nifti_tool','-mod_hdr',
               '-mod_field','vox_offset','352',
               '-in',tmpfilenii2,
               '-prefix',tmpfilenii3]
    sub.call(voffcmd)

    smimg = nib.load(tmpfilenii3)
    smimgdata = smimg.get_fdata()
    for rmfile in [tmpfilenii1,tmpfilenii2,tmpfilenii3,
                   tmpfilemnc1,tmpfilemnc2,
                   tmpfileniimask, tmpfilemncmask] :
        os.remove(rmfile)
    os.rmdir(tmpdir)
    return smimgdata
