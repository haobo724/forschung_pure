import sys, os
sys.path.insert(0, '..')

import glob, pydicom, copy
import MultiOrganSeg.project.data_module.helpers as helpers
import nibabel as nib
import numpy as np
import tqdm, h5py

import shutil

'''
Usage: To prepare the segmentation from Karthik

Date: Nov. 2020

One example of img/seg pair:
img: 102800_CT_Wb.nii.gz in F:\\Visceral\\visceral2-dataset\\retrieval-dataset\\CT_Volumes
seg: 102800_CT_Wb_seg_7.0.2004.201.mhd in I:\visceral_seg

'''

raw_path = "F:\\Visceral\\visceral2-dataset\\retrieval-dataset\\CT_Volumes"
seg_path = "I:\\visceral_seg"

# ds_path = "D:\\Data\\ct_data\\truncated_train"
ds_path = "D:\\Data\\ct_data\\visceral2-retrieval\\Train"

# moving average: -768.5109578274286 395.3051532206404
ds_mean = -769
ds_std = 395

def map_ground_truth(seg):
    # map the ground truth accordingly
    res = np.zeros_like(seg)
    to_list = [None] * 13
    to_list[0] = [24,11,12,16,26,1,14,29,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81]
    to_list[1] = [13]
    to_list[2] = [17,19,20,22,2,18,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,55,21]
    to_list[3] = [28,54,]
    to_list[4] = [9, 15]
    to_list[5] = [27]
    to_list[6] = [2,3,4,]
    to_list[7] = [10, ]
    to_list[8] = [5,6,7,8]
    to_list[9] = []
    to_list[10] = []
    to_list[11] = []
    to_list[12] = [23,25]

    for i in range(13):
        if len(to_list[i]) > 0:
            for j in to_list[i]:
                res[seg==j] = i

    return res

"""deprecated""" 
def convert_composite_nifti(nii_path, to_dir):
    # step 1: nii composite read
    # step 2: save slices to directory as H5py(easy access in the training)

    # step 1 & 2
    global ds_mean, ds_std
    nii_obj = nib.load(nii_path)
    nii_pixel_array = nii_obj.get_fdata()

    for i in range(nii_pixel_array.shape[0]):
        # if ds_mean == 0:
        #     ds_mean = np.mean(nii_pixel_array[:, :, i])
        #     ds_std = np.std(nii_pixel_array[:, :, i])
        # else:
        #     ds_mean = 0.999*ds_mean + 0.001*np.mean(nii_pixel_array[:, :, i])
        #     ds_std = 0.999*ds_std + 0.001*np.std(nii_pixel_array[:, :, i])

        f = h5py.File(os.path.join(to_dir, '{0:07d}.hdf5'.format(i)), 'w')
        # print(nii_obj.header.get_data_dtype())
        dset = f.create_dataset('pixel_array',
                                data=(nii_pixel_array[:, :, i].T - ds_mean) / ds_std,
                                dtype=nii_obj.header.get_data_dtype())
        dset.attrs['dtype'] = nii_obj.header.get_data_dtype().name
        dset.attrs['slice_num'] = i
        f.close()
    return 0


def convert_mhd_to_nifti(mhd_path, raw_path, tag):
    seg = helpers.mhd_imread(mhd_path)

    # map the labels
    seg = map_ground_truth(seg)
    # print(np.unique(seg))

    # transpose
    seg = seg.transpose(2,1,0)
    seg = np.flip(seg, 0)

    # print(seg.shape)
    
    # set seg pixdim to image pixdim
    raw_nifti = nib.load(raw_path)
    # pixdim = raw_nifti.header['pixdim']
    vol = nib.Nifti1Image(seg, raw_nifti.affine)
    vol.header['pixdim'] = raw_nifti.header['pixdim']
    nib.save(vol, os.path.join(ds_path, tag + "_seg.nii.gz"))

    return 0

# Get all mhd headers in seg_path
mhd_headers = glob.glob(os.path.join(seg_path, '*.mhd'))

# for debug
# mhd_headers = mhd_headers[:3]

# Parse the image tag and generate the img fname in raw_path
raw_fnames = dict()
seg_fnames = dict()
tags = []

for fname in mhd_headers:
    tag = os.path.basename(fname).split('_')[0]
    tags.append(tag)
    seg_fnames[tag] = fname
    raw_fnames[tag] = os.path.join(raw_path, tag + '_CT_Wb.nii.gz')

# Reformat the raw image and seg, then dump to ds_path
pbar = tqdm.tqdm(total=len(tags))
for tag in tags:
    # create dir in ds_path
    # to_path = os.path.join(ds_path, tag, tag)
    # if os.path.exists(to_path) is False:
    #     os.makedirs(to_path)

    # read the scanning in dicom and save to slice series.
    # convert_composite_nifti(raw_fnames[tag], to_path)

    # move scanning from external directory to disk
    shutil.copyfile(raw_fnames[tag], os.path.join(ds_path, tag + '_ct.nii.gz'))

    # read label from mhd and save to 'TAG_seg.nii.gz'
    convert_mhd_to_nifti(seg_fnames[tag], raw_fnames[tag], tag)

    pbar.update(1)

pbar.close()
print('finish')
print(ds_mean, ds_std)


