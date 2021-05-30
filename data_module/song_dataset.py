#test_dataloader 等是pl.module的默认hook

import data_module.custom_transform as custom_transform
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pytorch_lightning as pl
import logging
import monai.transforms as mxform
from monai.data.utils import list_data_collate
import numpy as np
import monai
from torch.utils.data import DataLoader
import torch
import glob
from abc import abstractmethod
import matplotlib.pyplot as plt
from ds import climain
'''
Some information about the date:
pixdim: [0.89, 0.89, 1.5]

#     "left_lung": 6,
#     "liver": 7,
#     "right_lung": 8,

'''

label_list = ['bg', 'liver', 'left_lung', 'right_lung']

# Base class of Visceral_dataset_2d/3d
# @pl.data_loader
#Todo:基类同visceral
class Song_dataset(pl.LightningDataModule):
    def __init__(self, data_folder, worker, batch_size,mode, **kwargs):
        super().__init__()

        # Get list of paths to all image and labels
        self.images, self.labels = self.get_image_and_label_path(data_folder)
        self.datafolder=data_folder

        # Configure logger
        self.tlogger = logging.getLogger(__name__)
        self.tlogger.info(f"Data folder: {data_folder}")

        # Save args
        self.worker = worker
        self.batch_size = batch_size

        # select mode
        self.mode=mode
    def setup(self, stage: str = None):
        self.keys = ("image", "label")
        # if stage == 'train':
        # n_train = int(len(self.images) * 0.8)
        # n_train = len(self.images)
        n_train_small = int(len(self.images) * 0.05)
        n_val = int(0.2 * n_train_small)

        # val should be at least one
        if n_val == 0:
            n_val += 1
            n_train_small -= 1

        # Log the basics of the dataset.
        self.tlogger.info(f"Number of training dataset: {n_train_small - n_val}")
        self.tlogger.info(f"Number of validation dataset: {n_val}")
        self.train_imgs = [
            {self.keys[0]: img, self.keys[1]: seg} for img, seg in
            zip(self.images[:n_train_small - n_val], self.labels[:n_train_small - n_val])
        ]
        self.val_imgs = [
            {self.keys[0]: img, self.keys[1]: seg} for img, seg in
            zip(self.images[n_train_small - n_val:n_train_small], self.labels[n_train_small - n_val:n_train_small])
        ]


    @staticmethod
    @abstractmethod
    def get_image_and_label_path(data_folder):
        pass

    @staticmethod
    @abstractmethod
    def get_xform(mode="train", keys=("image", "label")):
        pass

    def train_dataloader(self, cache=True):
        train_transform = self.get_xform(mode="train")

        cache_dir = None
        if cache:
            cache_dir = self.cache_dir

        train_ds = monai.data.PersistentDataset(
            data=self.train_imgs,
            transform=train_transform,
            cache_dir=cache_dir
        )
        print(type(train_ds))

        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.worker,
            pin_memory=torch.cuda.is_available(),
            collate_fn=list_data_collate
        )

        return train_loader

    def val_dataloader(self, cache=True):
        val_transform = self.get_xform(mode="val")
        cache_dir = None
        if cache:
            cache_dir = self.cache_dir

        val_ds = monai.data.PersistentDataset(
            data=self.val_imgs,
            transform=val_transform,
            cache_dir=cache_dir
        )
        val_loader = DataLoader(
            dataset=val_ds,
            batch_size=1,
            num_workers=self.worker,
            pin_memory=False,
            collate_fn=list_data_collate
        )
        return val_loader



# @pl.data_loader
class Visceral_dataset_2d(Song_dataset):
    def __init__(self, data_folder, worker, batch_size, **kwargs):
        super().__init__(data_folder, worker, batch_size, **kwargs)
        self.cache_dir = kwargs.get("cache_dir", ".\\tmp\\visceral-2d")
    @staticmethod
    def get_image_and_label_path(data_folder):
        images = sorted(glob.glob(os.path.join(data_folder, '*_ct.nii.gz')))
        labels = sorted(glob.glob(os.path.join(data_folder, '*_seg.nii.gz')))
        assert len(images) == len(labels)
        return images, labels

    @staticmethod
    def get_xform(mode="train", keys=("image", "label")):
        xforms = [
            mxform.LoadImaged(keys),
            custom_transform.Transposed(keys, (1, 0)),
            mxform.AddChanneld(keys),
            custom_transform.NormalizeLabeld(keys=['label'], from_list=[0,7, 8, 6], to_list=[0,1, 2, 3]),
            mxform.ScaleIntensityRanged(keys[0], a_min=-1024., a_max=3000., b_min=-1, b_max=1, clip=True),
            # mxform.Spacingd(keys, pixdim=[0.89, 0.89, 1.5], mode=("bilinear", "nearest"))
            # mxform.Resized(keys, spatial_size=(256,256), mode='nearest')
            mxform.SpatialPadd(keys, spatial_size=(512, 512), mode="reflect"),
            mxform.CenterSpatialCropd(keys, roi_size=[512, 512]),
        ]
        if mode == "train":
            xforms.extend([
                mxform.RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(-0.05, 0.05),
                    scale_range=(-0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                # mxform.RandSpatialCropd(keys, roi_size=[192, 192, 1], random_size=False),
                # mxform.RandSpatialCropSamplesd(keys, roi_size=[-1, -1, 1], num_samples=10, random_size=False),  # random_size=False?
                # mxform.SqueezeDimd(keys, -1)
            ])
            # dtype = (np.float32, np.uint8)
            dtype = (np.float32, np.float32)
        elif mode == "val":
            dtype = (np.float32, np.float32)
        elif mode == "infer":
            xforms = xforms[:-2]
            dtype = (np.float32, )
        xforms.extend([
            mxform.CastToTyped(keys, dtype=dtype),
            mxform.ToTensord(keys)
        ])
        return mxform.Compose(xforms)

    def val_dataloader(self, cache=True):
        val_transform = self.get_xform(mode="val")
        cache_dir = None
        if cache:
            cache_dir = self.cache_dir

        val_ds = monai.data.PersistentDataset(
            data=self.val_imgs,
            transform=val_transform,
            cache_dir=cache_dir
        )
        val_loader = DataLoader(
            dataset=val_ds,
            batch_size=5,
            num_workers=self.worker,
            pin_memory=False,
            collate_fn=list_data_collate
        )
        return val_loader

class Song_dataset_2d_with_CacheDataloder(Song_dataset):
    def __init__(self, data_folder, worker, batch_size,mode, **kwargs):
        super().__init__(data_folder, worker, batch_size,mode, **kwargs)
        self.cache_dir = None
    @staticmethod
    def get_image_and_label_path(data_folder):
        images = sorted(glob.glob(os.path.join(data_folder, '*_ct.nii.gz')))
        labels = sorted(glob.glob(os.path.join(data_folder, '*_seg.nii.gz')))
        assert len(images) == len(labels)
        return images, labels

    @staticmethod
    def get_xform(mode="train", keys=("image", "label")):
        pass


    def setup(self, stage: str = None):
        self.keys = ("image", "label","leaky")




    def train_dataloader(self,cache=None):
        """

                 MODE 1: ALL labeled patients --->30
                 MODE 2: ALL labeled patients --->15
                 MODE 3: ALL labeled patients ---10 + NoLiver ---10 + NoLung ---10 --->30
                 MODE 4: ALL labeled patients ---10 + NoLiver ---10 + NoLung ---10 --->30 with modifiy label/trainstep

                   """
        train_ds_alllabel,train_Nolung_patient_DS,train_NoLiver_patient_DS=climain(data_path=self.datafolder,
                                                                          Input_worker=self.worker,mode='train',
                                                                          dataset_mode=self.mode)

        if self.mode==1 or self.mode==2:
            train_ds=train_ds_alllabel
        else:
            train_ds=train_ds_alllabel+train_Nolung_patient_DS+train_NoLiver_patient_DS



        train_loader = monai.data.DataLoader(
            train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.worker,
            pin_memory=torch.cuda.is_available(),
            collate_fn=list_data_collate
        )
        return train_loader

    def val_dataloader(self,cache=None):
        # val_transform = self.get_xform(mode="val")
        # val_ds = monai.data.CacheDataset(
        #     data=self.val_imgs,
        #     transform=val_transform,
        # )
        val_ds_all,val_Nolung_patient_DS,val_NoLiver_patient_DS=climain(data_path=self.datafolder,
                                                                    Input_worker=self.worker,mode='val'
                                                                    ,dataset_mode=self.mode)
        if self.mode==1 or self.mode==2:
            val_ds=val_ds_all
        else:
            val_ds=val_ds_all+val_Nolung_patient_DS+val_NoLiver_patient_DS


        val_loader =  monai.data.DataLoader(
            dataset=val_ds,
            batch_size=6,
            num_workers=self.worker,
            pin_memory=False,
            collate_fn=list_data_collate
        )
        return val_loader





if __name__ == "__main__":
    # from helpers import scroll_slices_and_seg

    # dm = Visceral_dataset_2d(
    #     data_folder=r'F:\Forschung\multiorganseg\data\train_2D',
    #     # data_folder='E:\\data\\visceral2D',
    #     # data_folder='D:\\Data\\ct_data\\shuqing\\',
    #     worker=1,
    #     batch_size=1
    # )
    # # dm.setup(stage='train')
    #
    # loader = dm.train_dataloader(cache=False)
    # print(len(loader))
    # try:
    #     fig, axs = plt.subplots(1, 3)
    #     plt.ion()
    #     for item in loader:
    #         image = item["image"].numpy()
    #         label = item["label"].numpy()
    #         print(image.shape, label.shape,
    #               ' range: img [{}, {}], seg [{}, {}]'.format(np.amin(image), np.amax(image), np.amin(label),
    #                                                           np.amax(label)))
    #
    #         axs[0].imshow(image[0, 0,  ...])
    #         axs[1].imshow(label[0, 0, ...])
    #         axs[2].imshow(label[0, 0,  ...] * 0.5 + image[0, 0, ...] * 0.5)
    #         axs[0].set_title('image')
    #         axs[1].set_title('label')
    #         axs[2].set_title('image+label')
    #
    #         plt.pause(0.1)
    # except KeyboardInterrupt:
    #     print('Test end')
    #        val_ds,_,_=climain(data_path= self.datafolder,Input_worker=self.worker,mode='train')

        # loader = dm.val_dataloader(cache=False)
    # for item in loader:
    #     image = item["image"].numpy()
    #     label = item["label"].numpy()
    #     print(image.shape, label.shape)
    print("Test ends.")