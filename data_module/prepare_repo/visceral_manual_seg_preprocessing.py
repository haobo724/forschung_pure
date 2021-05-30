from mostoolkit.data_module.prepare_repo.helpers import convert_3D_to_2D, get_ds_ids
import numpy as np
import nibabel as nib
import glob
import os
import shutil
import matplotlib.pyplot as plt
from mostoolkit.data_module.visceral_dataset import label_dict
from mostoolkit.data_module.visceral_dataset import *

def pixdim_match(ori_path):
    # get the tags 
    fnames = glob.glob(os.path.join(ori_path, '*.nii.gz'))
    assert len(fnames) > 0

    fnames = [os.path.basename(i) for i in fnames]
    tags = [i.split('_')[0] for i in fnames]
    tags = list(set(tags))

    res_path = os.path.join(ori_path, 'res')
    for tag in tags:
        ct_fname = tag + '_ct.nii.gz'
        seg_fname = tag + '_seg.nii.gz'

        ct_vol = nib.load(os.path.join(ori_path, ct_fname))
        seg_vol = nib.load(os.path.join(ori_path, seg_fname))
        
        vol = nib.Nifti1Image(seg_vol.get_fdata(), ct_vol.affine)
        vol.header['pixdim'] = ct_vol.header['pixdim']

        nib.save(vol, os.path.join(res_path, seg_fname))

    print('done')

def label_correct(ori_path):
    fnames = glob.glob(os.path.join(ori_path, '*.nii.gz'))
    assert len(fnames) > 0

    fnames = [os.path.basename(i) for i in fnames]
    tags = [i.split('_')[0] for i in fnames]
    tags = list(set(tags))

    res_path = os.path.join(ori_path, 'res')
    for tag in tags:
        ct_fname = tag + '_ct.nii.gz'
        seg_fname = tag + '_seg.nii.gz'

        ct_vol = nib.load(os.path.join(ori_path, ct_fname))
        seg_vol = nib.load(os.path.join(ori_path, seg_fname))
        
        # correct the wrong label
        seg_data = seg_vol.get_fdata()
        seg_data[seg_data == 13] = 1

        vol = nib.Nifti1Image(seg_data, ct_vol.affine)
        vol.header['pixdim'] = ct_vol.header['pixdim']

        nib.save(vol, os.path.join(res_path, seg_fname))

    print('done')


# Create ground truth from multiple raw data
def raw_imread(path, size, dtype=np.uint8):
    # Size = (width, height) of the bytes-img. nslice will be inferred from the data
    # Return: raw_img.shape = (nshape, width, height)

    # read the raw bytes into 1d numpy buffers
    raw_buf = open(path, 'rb').read()
    raw_buf = np.frombuffer(raw_buf, dtype=dtype)

    # determine the number of slices
    nslice = int(len(raw_buf) / np.prod(size))
    size.insert(0, nslice)

    # reshape buf to image size
    raw_img = raw_buf.reshape(size)
    size.pop(0)
    return raw_img

# Filenames helpers 
def get_id_list(dir):
    return os.listdir(dir)

def get_groundtruth_path(dir, id):
    return os.path.join(dir, id, 'label')

def get_save_fname(id):
    # return the saving fname of ground truth
    return os.path.join(groundtruth_path, '{}_seg.nii.gz'.format(id))

def generate_composite_label(dir, id, size=[512, 512]):
    # read all nii and raw label in dir and generate composite label according to organ label
    # and save using nib

    # create the container
    size_ = size
    label_vol = None
    affine = np.eye(4)
    # Get all nii and raw fnames
    print('\ngenerating composite gt of {}'.format(id))
    for l in label_dict:
        seg_path = glob.glob(os.path.join(dir, '*_{}.nii.gz'.format(l)))
        if len(seg_path) == 0:
            seg_path = glob.glob(os.path.join(dir, '*_{}.nii'.format(l)))

        if len(seg_path) == 0:
            # The ground truth in .raw
            seg_path = glob.glob(os.path.join(dir, '*_{}.raw'.format(l)))
            if len(seg_path) == 0:
                # missing segmentation
                continue
            gt = raw_imread(seg_path[0], size=size_, dtype=np.int8)
            gt = np.clip(gt, 0, 1).transpose(2, 1, 0)
        else:
            # The ground truth in .nii.gz
            gt = nib.load(seg_path[0])
            affine = gt.affine
            gt = np.clip(gt.get_fdata(), 0, 1)
        
        if label_vol is None:
            label_vol = np.zeros_like(gt)
        
        label_vol[gt==1] = label_dict[l]
        print('{} added.'.format(l))
    
    # save as composite label
    vol = nib.Nifti1Image(label_vol.astype(np.uint8), affine)
    nib.save(vol, get_save_fname(id))
    print('{} done'.format(id))


if __name__ == "__main__":
    # groundtruth_path = 'D:\\Data\\ct_data\\visceral_manual_seg\\gt_raw'
    # if os.path.exists(groundtruth_path) is False:
    #     os.makedirs(groundtruth_path)

    # dir_ = 'D:\\Data\\ct_data\\visceral_manual_seg'
    # # pixdim_match(dir_)
    # label_correct(dir_)

    # Generation of 2D dataset
    # raw_dir = 'E:\\data\\visceral_raw'
    raw_dir = 'D:\\Data\\ct_data\\visceral_manual_seg\\train_3D'
    tar_dir = 'D:\\Data\\ct_data\\visceral_manual_seg\\train_2D'

    # vol_fnames = sorted(glob.glob(os.path.join(raw_dir, '*_ct.nii.gz')))
    # seg_fnames = sorted(glob.glob(os.path.join(raw_dir, '*_seg.nii.gz')))
    
    # shutil.rmtree(tar_dir)
    ids = get_ds_ids(raw_dir)
    convert_3D_to_2D(ids, raw_dir, tar_dir)

    # debug of reading raw ground truth.
    # path = "E:\\data\\visceral_manual_seg\\data\\10000005\\label\\10000005_skin.raw"
    # size = [512, 512]
    # label_image = raw_imread(path, size, dtype=np.int8)

    # plt.imshow(label_image[0])
    # plt.show()

    # print(label_image.shape)

    # redo the ground truth
    # path = 'F:\\ct_data\\data'
    # ids = get_id_list(path)

    # id = ids[-1]
    # gt_path = get_groundtruth_path(path, id)
    # generate_composite_label(gt_path, id)
