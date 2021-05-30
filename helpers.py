import logging
from argparse import ArgumentParser
import nibabel as nib
import numpy as np

import glob
import os

"""
Logging helpers
"""
def logging_init(log_fname, log_lvl=logging.INFO):
    logging.basicConfig(format='%(asctime)s [%(levelname)-5.5s] %(message)s',
                        level=log_lvl,
                        handlers=[
                            logging.FileHandler(log_fname, mode='w'),
                            logging.StreamHandler()
                        ])


def logging_args_config(logger):
    pass

"""
training helpers
"""
def add_training_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--data_folder', nargs='+', type=str, default='/cluster/liu/data/')
    parser.add_argument('--cache_dir', type=str, default="/cluster/liu/project/tmp/monai")
    # parser.add_argument("--log_path", type=str, default='train')
    parser.add_argument('--cluster', type=bool, default=False)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_data_folder", type=str, default='/cluster/liu/data/visceral_manual_seg/test')

    return parser

def add_inference_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--ckpt_dir', type=str, default=None, help='Path to the model checkpoint, regularly saved in ./lightning_logs/..')
    parser.add_argument('--raw_dir', type=str, default=None, help='Path to the directory of the nifti image files.')
    parser.add_argument('--cuda', type=bool, default=True, help='True, if GPU accelerator is available.')

    return parser

"""
file manipulation helper
"""
def get_ckpts_paths(version):
    ckpts = ['.\\lightning_logs\\version_{}\\final.ckpt'.format(version)]
    ckpts.extend(glob.glob('.\\lightning_logs\\version_{}\\checkpoints\\*.ckpt'.format(version)))
    return ckpts

"""
Evaluation helper
"""
def get_fname_with_suffix(fname, suffix):
    # example 10000081_ct.nii.gz
    basename = fname.split('.')[0]
    old_suffix = fname[len(basename):]
    return basename + '_' + suffix + old_suffix


def get_model_id(model):
    base = os.path.basename(model)
    return base[:-(len(base.split('.')[-1]) + 1)]

def get_version(model):
    # get verision from ckpt dir
    # print(model)
    if type(model) is not str:
        ValueError('Model should only be path')
    dir = model.split(os.sep)
    # print(dir)
    if dir[-1] == 'final.ckpt':
        version_str = dir[-2]
        # print(version_str)
    else:
        version_str = dir[-3]
    version = version_str.split('_')[-1]
    return int(version)

def scale_intensity_range(img, amin, amax, bmin, bmax):
    # scale the img intensity from (amin, amax) to (bmin, bmax)
    # resemble monai.transforms.ScaleIntensityRange(...)

    img = (img - amin) / (amax - amin)
    img = img * (bmax - bmin) + bmin
    return img

def create_confusion_matrix(gt, eval_vol, num_cls):
    # flatten the inputs
    gt = np.array(gt).flatten()
    eval_vol = np.array(eval_vol).flatten()

    confusion = np.zeros([num_cls, num_cls])
    # confusion[i, j]: num of pixels of class i, predicted as class j
    for i in range(num_cls):
        for j in range(num_cls):
            confusion[i, j] = np.sum(np.logical_and(gt == i, eval_vol == j))
    return confusion.astype(np.int32)


def MOS_eval(pred_path, gt_path, label_list):
    gt = nib.load(gt_path).get_fdata()
    eval_vol = nib.load(pred_path).get_fdata()

    # print(gt.shape)
    # print(eval_vol.shape)

    # Caution, class 13 is class 1
    gt[gt == 13] = 1

    # Since the ground truth is always smaller, crop the eval_vol according to ground truth
    num_slices = gt.shape[-1]

    eval_vol = eval_vol[:, :, :num_slices]

    print('')
    print('Creation of confusion matrix in progress: ')
    confusion = create_confusion_matrix(gt, eval_vol, len(label_list))
    # print(confusion)

    # classwise recall and precision
    eps = 1e-4
    recall = np.diagonal(confusion) / (np.sum(confusion, axis=1) + eps)
    precision = np.diagonal(confusion) / (np.sum(confusion, axis=0) + eps)
    dice = 2 * (recall * precision) / (recall + precision + eps)

    print('')
    name_msg = ''
    for l in label_list:
        name_msg += l.ljust(10)
    print(name_msg)

    print()
    recall_msg = ''
    for r in recall:
        recall_msg += '{0:.2g}%  |'.format(r * 100).ljust(10)
    print('|**Recall**|', recall_msg)

    precision_msg = ''
    for p in precision:
        precision_msg += '{0:.2g}%  |'.format(p * 100).ljust(10)
    print('|**Precision**|', precision_msg)

    dice_msg = ''
    for d in dice:
        dice_msg += '{0:.2g}%  |'.format(d * 100).ljust(10)
    print('|**dice**|', dice_msg)


if __name__ == '__main__':
    atlas_seg_path = 'D:\\Chang\\MultiOrganSeg\\mostoolkit\\OrganPositionEmbedding\\Model_output\\atlas_seg.nii.gz'
    gt_seg_path = 'D:\\Data\\ct_data\\visceral_manual_seg\\10000081_seg.nii.gz'

    MOS_eval(atlas_seg_path, gt_seg_path)
