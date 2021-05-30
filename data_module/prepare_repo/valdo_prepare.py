import os, glob
import nibabel as nib
import shutil
import matplotlib.pyplot as plt
import numpy as np

def get_id_dict(dir):
    fnames = glob.glob(os.path.join(dir, '*.nii.gz'))
    ids = [get_sub_id(i) for i in fnames]
    return dict(zip(ids, fnames))

def get_sub_id(fname):
    return os.path.basename(fname).split('_')[0].split('-')[-1]

def parse_valdo_sub_file(fnames):
    tags = [(f.split('-')[-1]).split('.')[0] for f in fnames]
    fname_dict = dict(zip(tags, fnames))
    return fname_dict
    # return (fname_dict.get('masked_T1', None),
    #         fname_dict.get('masked_T2', None),
    #         fname_dict.get('masked_FLAIR', None),
    #         fname_dict.get('masked_Regions', None), 
    #         fname_dict.get('Rater1_PVSSeg', None),
    #         fname_dict.get('Rater2_PVSSeg', None))

def copy_from_sub_to_task1_3D(ori_path, tar_path_3D, categories):
    # walk through the ori path and get all "sub-*"
    subs = glob.glob(os.path.join(ori_path, 'sub-*'))
    # print(subs)
    for sub in subs:
        # get all nifti
        fnames = glob.glob(os.path.join(sub, 'sub-*.nii.gz'))
        # t1, t2, flair, region, r1seg, r2seg = parse_valdo_sub_file(fnames)
        print(fnames)
        sub_dict = parse_valdo_sub_file(fnames)
        # print(sub_dict)
        # copy file to tar_path
        for c in categories:
            f = sub_dict.get(c, None)
            if f is not None:
                shutil.copy(f, os.path.join(tar_path_3D, c))
                print('{} copied.'.format(f))
    print('done.')

def generate_task1_2D_ds(path_3D, path_2D, categories):
    rater1_ids = [get_sub_id(i) for i in glob.glob(os.path.join(path_3D, categories[-2], '*.nii.gz'))]
    for c in categories[:-2]:
        fdict = get_id_dict(os.path.join(path_3D, c))
        for id in rater1_ids:
            convert_to_2D(fdict[id], os.path.join(path_2D, c), 'mri')
            print('{} converted'.format(fdict[id]))
    for c in categories[-2:]:
        fdict = get_id_dict(os.path.join(path_3D, c))
        for id in rater1_ids:
            convert_to_2D(fdict[id], os.path.join(path_2D, c), 'seg', np.uint8)
            print('{} converted'.format(fdict[id]))

def convert_to_2D(vol_fname, tar_path, suffix='mri', dtype=np.float32):
    vol = nib.load(vol_fname)
    vol_img = vol.get_fdata().astype(dtype)

    vol_id = get_sub_id(vol_fname)
    vol_img = vol_img.transpose(1, 0, 2)
    for i in range(vol_img.shape[0]):
        slice = vol_img[i, ...]
        # plt.imshow(slice)
        # plt.show()

        slice = nib.Nifti1Image(slice, vol.affine)
        slice.header['pixdim'] = vol.header['pixdim']

        nib.save(slice, os.path.join(tar_path, '{}_{}_{}.nii.gz'.format(vol_id, i, suffix)))
    
    # print('{} done'.format(vol_id))


if __name__ == '__main__':
    ori_path = 'D:\\Data\\valdo\\Task1.tar\\Task1'
    tar_path_3D = 'D:\\Data\\valdo\\Task1.tar\\task1_3D'
    tar_path_2D = 'D:\\Data\\valdo\\Task1.tar\\task1_2D'

    # categorize MRI volumes to [t1, t2, flair, region, rater1_pvsseg, rater2_pvsseg]
    # categories = ['t1', 't2', 'flair', 'region', 'rater1_pvsseg', 'rater2_pvsseg']
    categories = ['masked_T1', 'masked_T2', 'masked_FLAIR', 'masked_Regions', 'Rater1_PVSSeg', 'Rater2_PVSSeg']
    
    for c in categories:
        if os.path.exists(os.path.join(tar_path_3D, c)) is False:
            os.makedirs(os.path.join(tar_path_3D, c))

    # copy_from_sub_to_task1_3D(ori_path, tar_path_3D, categories)

    # convert 3D dataset to 2D dataset
    for c in categories:
        if os.path.exists(os.path.join(tar_path_2D, c)) is False:
            os.makedirs(os.path.join(tar_path_2D, c))
    generate_task1_2D_ds(tar_path_3D, tar_path_2D, categories)









