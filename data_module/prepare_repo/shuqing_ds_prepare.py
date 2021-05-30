import os, glob
import nibabel as nib
import shutil
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import tqdm
from mostoolkit.data_module.prepare_repo.helpers import *

# cvt_dict = {
#     1000: 0,
#     2000: 0,
#     3000: 7,
#     4000: 0,
# }

cvt_dict = {
    1: 0,
    2: 0,
    3: 7,
    4: 0,
    5: 8,
    6: 6,
    7: 0,
    8: 0,
    9: 0,
}

def copy_ct_to_tar_3D(ids, ct_dict, tar_path_3D):
    for id in ids:
        shutil.copy(ct_dict[id], os.path.join(tar_path_3D, id + '_ct.nii.gz'))
        print(id, ' ct copied')

def copy_seg_to_tar_3D(ids, seg_dict, tar_path_3D):
    for id in ids:
        # shutil.copy(seg_dict[id], os.path.join(tar_path_3D, id+'_seg.nii.gz')])
        # label_correct(os.path.join(tar_path_3D, id+'_seg.nii.gz'))

        # label correct needed
        vol = nib.load(seg_dict[id])
        vol_data = vol.get_fdata()
        vol_tmp = np.zeros_like(vol_data)
        for ori in cvt_dict:
            vol_tmp[vol_data == ori] = cvt_dict[ori]
        tar = nib.Nifti1Image(vol_tmp, vol.affine)
        tar.header['pixdim'] = vol.header['pixdim']
        nib.save(tar, os.path.join(tar_path_3D, id + '_seg.nii.gz'))
        print(id, ' seg copied')

# temp func
def split_test_set(train_3D_dir):
    # manually split the testing ds from training ds
    ids = get_ds_ids(train_3D_dir)
    test_ids = ids[-8:]
    train_ids = ids[:-8]

    test_dir = 'D:\\Data\\ct_data\\shuqing\\test'
    # move test ds to test dir and delete the initial one
    for id in test_ids:
        f_ct = os.path.join(train_3D_dir, id + '_ct.nii.gz')
        f_seg = os.path.join(train_3D_dir, id + '_seg.nii.gz')
        shutil.copy(f_ct, test_dir)
        shutil.copy(f_seg, test_dir)
        os.remove(f_ct)
        os.remove(f_seg)
        print('{} done'.format(id))

# before: 38,064
# temp func
def remove_test_ds(train_2D_dir, test_dir):
    ids = get_ds_ids(test_dir)

    # remove all slices of ids in train_2D_dir
    for id in ids:
        fs = glob.glob(os.path.join(train_2D_dir, '{}_*'.format(id)))
        for f in fs:
            os.remove(f)
            print('{} removed.'.format(os.path.basename(f)))

if __name__ == '__main__':
    # ct_path = 'F:\\DECT2\\NIIOriginal'
    # seg_path = 'F:\\DECT2\\DLMask_9'

    # tar_path_3D = 'D:\\Data\\ct_data\\shuqing\\data_3D'
    # tar_path_2D = 'D:\\Data\\ct_data\\shuqing\\data_2D'
    # if os.path.exists(tar_path_2D) is False:
    #     os.makedirs(tar_path_2D)
    # if os.path.exists(tar_path_3D) is False:
    #     os.makedirs(tar_path_3D)

    # # get keys of seg and volume
    # ids = get_ds_ids(seg_path)
    # seg_dict = get_seg_dict(ids, seg_path)
    # ct_dict = get_ct_dict(ids, ct_path)

    # # copy seg and volume to tar_path_3D
    # copy_ct_to_tar_3D(ids, ct_dict, tar_path_3D)
    # copy_seg_to_tar_3D(ids, seg_dict, tar_path_3D)

    # # convert 3D data to 2D data
    # convert_3D_to_2D(ids, tar_path_3D, tar_path_2D)

    # resample vols using normalized pixdim
    # ori_path_3D = 'D:\\Data\\ct_data\\visceral_manual_seg'
    # tar_path_3D = 'D:\\Data\\ct_data\\visceral_manual_seg\\train_3D'
    # tar_path_2D = 'D:\\Data\\ct_data\\visceral_manual_seg\\train_2D'
    # pixdim = [0.89, 0.89, 1.5]
    # resample ct vol
    # dataset_resample(ori_path_3D, tar_path_3D, pixdim)

    # Generate 2D dataset
    # ids = get_ds_ids(tar_path_3D)
    # convert_3D_to_2D(ids, tar_path_3D, tar_path_2D)

    train_2D_dir = 'D:\\Data\\ct_data\\shuqing\\data_2D_unit_pd'
    train_3D_dir = 'D:\\Data\\ct_data\\shuqing\\data_3D_unit_pd'
    test_dir = 'D:\\Data\\ct_data\\shuqing\\test'
    # split_test_set(train_3D_dir)
    remove_test_ds(train_2D_dir, test_dir)
