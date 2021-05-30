import os, sys, logging, copy
import glob, json
import numpy as np
import nibabel as nib

import torch

import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import SimpleITK as sitk
import matplotlib as mpl
from skimage import feature

'''Helpers for checking datasets before training'''


def get_ct_fname(dir, frame):
    return os.path.join(dir, os.path.basename(dir), '{0:07d}'.format(frame))


def check_dataset_completeness(dir):
    # check whether {index}.dcm and GroundTruth.nii.gz is present
    index = os.path.basename(dir)
    completeness = True

    # check the presence of dicom
    if os.path.exists(os.path.join(dir, index)) is False:
        completeness = False
    elif len(os.listdir(os.path.join(dir, index))) == 0:
        completeness = False

    # check the presence of ground truth
    if os.path.exists(os.path.join(dir, 'GroundTruth.nii.gz')) is False:
        completeness = False

    return completeness


def dataset_check(ds_path):
    # check available dataset (vol and label in the json file)
    json_path = os.path.join(ds_path, 'dataset_index.json')
    if os.path.exists(json_path) is False:
        # create .json if it is not created
        ind = dict()
        ind['index'] = os.listdir(ds_path)

        checked = dict()
        checked['index'] = []
        checked['num_slices'] = list()
        # check the completeness of each item
        for i in ind['index']:
            if check_dataset_completeness(os.path.join(ds_path, i)) is True:
                checked['index'].append(i)
                # read out the number of slices
                num_slices = min(len(os.listdir(os.path.join(ds_path, i, i))),
                                 nib.load(ds_path + '\\' + i + '\\GroundTruth.nii.gz').get_fdata().shape[-1])
                checked['num_slices'].append(num_slices)

        with open(json_path, 'w') as f:
            json.dump(checked, f)
        print('No index was here. Index is now created.')
    else:
        # check whether new data is added to the dataset
        with open(json_path) as f:
            old_ind = json.load(f)
        new_ind = dict()
        new_ind['index'] = os.listdir(ds_path)

        # check the completeness
        checked = dict()
        checked['index'] = []
        checked['num_slices'] = list()
        for i in new_ind['index']:
            if check_dataset_completeness(os.path.join(ds_path, i)) is True:
                checked['index'].append(i)
                num_slices = min(len(os.listdir(os.path.join(ds_path, i, i))),
                                 nib.load(ds_path + '\\' + i + '\\GroundTruth.nii.gz').get_fdata().shape[-1])
                checked['num_slices'].append(num_slices)

        if old_ind['index'] == checked['index']:
            print("Dateset checked, nothing is new here.")
        else:
            with open(json_path, 'w') as f:
                json.dump(checked, f)
            print('Detected changes in the dataset. Index now updated.')


'''Helpers for creating dataset'''
def get_nii_fnames(dir):
    # helper to get all .nii and .nii.gz filenames in dir
    return glob.glob(os.path.join(dir, "*.nii.gz")) + glob.glob(os.path.join(dir, "*.nii"))


def get_raw_fnames(dir):
    return glob.glob(os.path.join(dir, "*.raw"))


def label_fname_parse(fname):
    # the filename is assumed to be "10000005_left_lung.nii.gz" or "10000005_bone_surface.raw"
    # 1, remove fname suffix
    fname_ = fname.split('.')[0]

    # 2, remove id and join the list with '_'
    organ_name = fname_.split('_')[1:]
    organ_name = '_'.join(organ_name)

    # 3. look up label_dict and return -1 if missing key
    label = label_dict.get(organ_name, -1)

    if label == -1:
        logging.warning(" %s is not in the dict.", fname)
        label = label_dict['bg']

    return label


def raw_imread(path, size, dtype=np.float32):
    # Size = (slice, width, height) of the bytes-img. Should be inspected using ImageJ

    # read the raw bytes into 1d numpy buffers
    raw_buf = open(path, 'rb').read()
    raw_buf = np.frombuffer(raw_buf, dtype=dtype)

    # reshape buf to image size
    raw_img = raw_buf[:np.prod(size)].reshape(size)

    # for laura's raw data, one slice contains 16 sub-slices
    #
    slice, width, height = size
    width = int(width / 4)
    height = int(height / 4)
    raw_img = raw_img.reshape(slice, 4, height, width * 4)
    raw_img = np.swapaxes(raw_img, 2, 3)
    raw_img = raw_img.reshape(slice, 16, height, width)
    raw_img = raw_img.reshape(slice * 16, height, width)
    raw_img = np.swapaxes(raw_img, 0, 2)
    raw_img = np.swapaxes(raw_img, 0, 1)

    return raw_img


def mhd_imread(path, dtype=np.uint8):
    # read segmentation from .mhd format, using SimpleITK
    itkimage = sitk.ReadImage(path)
    seg = sitk.GetArrayFromImage(itkimage).astype(dtype=dtype)
    return seg

'''Helpers for image format transformation'''
def Dcm2Tensor(vol):
    # format the volume from dcmread() to the input tensor of the model
    vol = vol.astype(np.float32)
    vol = torch.from_numpy(vol)

    vol = vol.unsqueeze(1)
    return vol


'''Helpers for generating the distance map for lukas loss'''
def get_distance_map(label, sampling=(0.01, 0.01), weights_range=(0.0, 1.0)):
    # get distance map based on the label.
    mask = label == 0

    if max(sampling) == 0:
        # debug mode
        return np.ones_like(label)

    if len(label.shape) == 2:
        # 2D
        sampling = np.array(sampling) * np.pi
        distance_map = distance_transform_edt(mask, sampling=sampling)
        distance_map = np.cos(np.clip(distance_map - np.pi, -np.pi, 0.0)) * 0.5 + 0.5
        distance_map = distance_map + (~mask).astype(np.float32)
        distance_map = np.interp(distance_map, (distance_map.min(), distance_map.max()), list(weights_range))
    elif len(label.shape) == 3:
        # 3D, set the sampling value of the 3rd dimension a very big one, so the distance is in 2D plane
        sampling = (*sampling, 0.99)
        sampling = np.array(sampling) * np.pi
        distance_map = distance_transform_edt(mask, sampling=sampling)
        distance_map = np.cos(np.clip(distance_map - np.pi, -np.pi, 0.0)) * 0.5 + 0.5
        distance_map = distance_map + (~mask).astype(np.float32)
        distance_map = np.interp(distance_map, (distance_map.min(), distance_map.max()), list(weights_range))
    else:
        TypeError('Incorrect dimension of Label map, should be either (w, h), or (w, h, c)')
        distance_map = None
    return distance_map

"""
Helpers for viewer
"""
class IndexTracker(object):
    def __init__(self, ax, volume, segmentation=None, title="", show_contour=False):
        self.volume = volume
        self.segmentation = segmentation
        if len(volume.shape) == 4:
            rows, cols, channels, self.slices = volume.shape
        elif len(volume.shape) == 3:
            rows, cols, self.slices = volume.shape
        self.ind = self.slices // 2
        self.ax = ax
        self.title = title
        if segmentation is not None:
            self.im = self.ax[0].imshow(self.volume[:, :, self.ind], vmin=0, vmax=1)
            if title != "":
                self.ax[0].set_title(self.title)
            self.show_contour = show_contour
            if show_contour:
                self.contour = np.moveaxis(np.array(
                    [feature.canny(self.segmentation[..., slice_ind]) for slice_ind in range(segmentation.shape[-1])]),
                    0, -1) > 0.5
                self.cont = self.ax[0].imshow(self.contour[..., self.ind], cmap="Reds",
                                              alpha=1.0 * (self.contour[..., self.ind] > 0).astype(float))
            self.seg = self.ax[1].imshow(self.segmentation[:, :, self.ind])
        else:
            self.im = self.ax.imshow(self.volume[..., self.ind], vmin=0, vmax=1)
            if title != "":
                self.ax.set_title(self.title)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if self.segmentation is not None:
            self.im.set_data(self.volume[:, :, self.ind])
            self.seg.set_data(self.segmentation[:, :, self.ind])
            self.ax[0].set_ylabel('slice %s' % self.ind)
            if self.show_contour:
                self.cont.set_data(self.contour[..., self.ind])
                self.cont.set_alpha(1.0 * (self.contour[..., self.ind] > 0).astype(float))
            self.im.axes.figure.canvas.draw()
            self.seg.axes.figure.canvas.draw()
            if self.show_contour:
                self.cont.axes.figure.canvas.draw()
        else:
            self.im.set_data(self.volume[..., self.ind])
            self.ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()


def scroll_slices(volume, title=""):
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, volume, title="")
    fig.suptitle(title)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


def scroll_slices_and_seg(volume, segmentation, title="", show_contour=None):
    mpl.rc('image', cmap='gray')
    fig, ax = plt.subplots(1, 2)
    tracker = IndexTracker(ax, volume, segmentation, "", show_contour)
    fig.suptitle(title)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


if __name__ == '__main__':
    # # TEST: distance map generation
    # ori = np.zeros([512, 512, 2])
    # # add some pattern
    # ori[100:400, 100:400, 0] = 1.0
    # ori[200:500, 200:500, 1] = 1.0
    # fig, axes = plt.subplots(2, 6)
    # axes[0, 0].imshow(ori[..., 0])
    # axes[1, 0].imshow(ori[..., 1])
    # for i in range(1, 6):
    #     sampling = (i * 0.01 - 0.01, i * 0.01 - 0.01)
    #     dist_map = get_distance_map(ori, sampling)
    #     axes[0, i].imshow(dist_map[..., 0], vmin=0, vmax=1)
    #     axes[1, i].imshow(dist_map[..., 1], vmin=0, vmax=1)
    # plt.show()
    print(label_list[0] + '_seg')
