import os, glob
import nibabel as nib
import shutil
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import tqdm


def parse_id(fname):
    # Example of fname: 23000AIREI_Seg.nii.gz
    fname = os.path.basename(fname)
    return fname.split('_')[0]

def get_ds_ids(path):
    fnames = sorted(glob.glob(os.path.join(path, '*_seg.nii.gz')))
    ids = [parse_id(i) for i in fnames]
    return ids

def get_seg_dict(ids, seg_path):
    segs = sorted(glob.glob(os.path.join(seg_path, '*_seg.nii.gz')))
    return dict(zip(ids, segs))

def get_ct_dict(ids, ct_path):
    ct_ids = os.listdir(ct_path)
    ct_dict = dict()
    for id in ct_ids:
        # We only use first eV level
        ct_dict[id] = glob.glob(os.path.join(ct_path, id, '*_1.nii.gz'))[0]
        assert ct_dict[id] is not None
    return ct_dict

def _convert_3D_to_2D(vol_fname, tar_path, suffix='ct', dtype=np.float32, length=0):
    vol = nib.load(vol_fname)
    vol_img = vol.get_fdata().astype(dtype)

    vol_id = parse_id(vol_fname)
    # vol_img = vol_img.transpose(1, 0, 2)
    if length == 0:
        length = vol_img.shape[-1]
    
    for i in range(length):
        slice = vol_img[..., i]
        # plt.imshow(slice)
        # plt.show()

        slice = nib.Nifti1Image(slice, vol.affine)
        slice.header['pixdim'] = vol.header['pixdim']

        nib.save(slice, os.path.join(tar_path, '{}_{}_{}.nii.gz'.format(vol_id, i, suffix)))
    return vol_img.shape[-1]

def convert_3D_to_2D(ids, tar_path_3D, tar_path_2D):
    print('generating 2D dataset')
    pbar = tqdm.tqdm(total=len(ids))
    for id in ids:
        ct_fname = os.path.join(tar_path_3D, id + '_ct.nii.gz')
        seg_fname = os.path.join(tar_path_3D, id + '_seg.nii.gz')

        length = min(nib.load(ct_fname).get_fdata().shape[-1], nib.load(seg_fname).get_fdata().shape[-1])

        _convert_3D_to_2D(ct_fname, tar_path_2D, suffix='ct', dtype=np.float32, length=length)
        _convert_3D_to_2D(seg_fname, tar_path_2D, suffix='seg', dtype=np.uint8, length=length)
        pbar.update(1)
    pbar.close()

def sitk_resample(ori_path, tar_path, pixdim, interp='linear'):
    img = sitk.ReadImage(ori_path)
    ori_spacing = img.GetSpacing()
    ori_size = img.GetSize()
    ori_direction = img.GetDirection()
    ori_pixelid = img.GetPixelID()
    ori_origin = img.GetOrigin()

    new_size = [int(sz * spc / nspc) for sz, spc, nspc in zip(ori_size, ori_spacing, pixdim)]

    ref_img = sitk.Image(new_size, ori_pixelid)
    ref_img.SetDirection(ori_direction)
    ref_img.SetOrigin(ori_origin)
    ref_img.SetSpacing(pixdim)
    if interp == 'linear':
        img_out = sitk.Resample(img, ref_img, interpolator=sitk.sitkLinear)
    elif interp == 'nearest':
        img_out = sitk.Resample(img, ref_img, interpolator=sitk.sitkNearestNeighbor)
    else:
        ValueError('Unknown interpolation method: {}'.format(interp))
    sitk.WriteImage(img_out, tar_path)

def dataset_resample(ori_path_3D, tar_path_3D, pixdim):
    print('resampling 3D ct vols..')
    vol_3D = sorted(glob.glob(os.path.join(ori_path_3D, '*_ct.nii.gz')))
    pbar = tqdm.tqdm(total=len(vol_3D))
    for vol in vol_3D:
        tar_path = os.path.join(tar_path_3D, os.path.basename(vol))
        sitk_resample(vol, tar_path, pixdim)
        pbar.update(1)
    pbar.close()

    # resample ct seg
    print('resampling ct segs..')
    seg_3D = sorted(glob.glob(os.path.join(ori_path_3D, '*_seg.nii.gz')))
    pbar = tqdm.tqdm(total=len(vol_3D))
    for seg in seg_3D:
        tar_path = os.path.join(tar_path_3D, os.path.basename(seg))
        sitk_resample(seg, tar_path, pixdim, interp='nearest')
        pbar.update(1)
    pbar.close()