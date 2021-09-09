# test_dataloader 等是pl.module的默认hook

import data_module.custom_transform as custom_transform
import pytorch_lightning as pl
import monai.transforms as mxform
from monai.data.utils import list_data_collate
import numpy as np
from monai.data import DataLoader
import torch
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
class Song_dataset_2d_with_CacheDataloder(pl.LightningDataModule):
    def __init__(self, data_folder, worker, batch_size, mode,clean=False, **kwargs):
        super().__init__()
        self.cache_dir = None
        self.train_ds = None
        self.val_ds = None
        self.worker = worker
        self.datafolder = data_folder
        self.clean=clean
        self.batch_size = batch_size
        self.mode = mode

    @staticmethod
    def get_xform(mode="train", keys=("image", "label", "leaky"), leaky=None, leakylist=None):
        xforms = [
            mxform.LoadImaged(keys[:2]),
            custom_transform.Transposed(keys[:2], (1, 0)),
            mxform.AddChanneld(keys[:2]),
            custom_transform.NormalizeLabeld(keys=['label'], from_list=[0, 7, 8, 6], to_list=[0, 1, 2, 3]),
            # custom_transform.Leakylabel(keys=["leaky"]),
            mxform.ScaleIntensityRanged(keys[0], a_min=-1024., a_max=3000., b_min=-1, b_max=1, clip=True),
            # # mxform.Spacingd(keys, pixdim=[0.89, 0.89, 1.5], mode=("bilinear", "nearest"))
            # # mxform.Resized(keys, spatial_size=(256,256), mode='nearest')
            mxform.SpatialPadd(keys[:2], spatial_size=(512, 512), mode="edge"),
            mxform.CenterSpatialCropd(keys[:2], roi_size=[512, 512]),
        ]
        if mode == "train":
            xforms.extend([
                mxform.RandAffined(
                    keys[:2],
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
        elif mode == "test":
            xforms = xforms[:-2]
            dtype = (np.float32, np.float32)

        if leaky == 'liver':
            xforms.extend([
                custom_transform.Leakylabel(keys=["leaky"], leakylist=leakylist, leaky=leaky),
                custom_transform.NormalizeLabeld(keys=['label'], from_list=[0, 1, 2, 3], to_list=[0, 0, 2, 3]),

            ])
        elif leaky == 'lung':
            xforms.extend([
                custom_transform.Leakylabel(keys=["leaky"], leakylist=leakylist, leaky=leaky),
                custom_transform.NormalizeLabeld(keys=['label'], from_list=[0, 1, 2, 3], to_list=[0, 1, 0, 0]),

            ])
        elif leaky == 'all':
            xforms.extend([
                custom_transform.LeakylabelALLFALSE(keys=["leaky"]),
                # mxform.CastToTyped(keys[-1], dtype=torch.bool),

                # mxform.ToTensord(keys[-1]),

            ])
        xforms.extend([
            mxform.CastToTyped(keys[:2], dtype=dtype),
            mxform.ToTensord(keys[:2])
        ])
        return mxform.Compose(xforms)

    #     "liver": 1,
    #     "right_lung": 2,
    #     "left_lung": 3,

    def setup(self, stage: str = None):
        self.keys = ("image", "label", "leaky")

        train_ds_alllabel, train_Nolung_patient_DS, train_NoLiver_patient_DS = climain(data_path=self.datafolder,
                                                                                       Input_worker=self.worker,
                                                                                       mode='train',
                                                                                       dataset_mode=self.mode,
                                                                                       clean=self.clean)
        val_ds_alllabel, val_Nolung_patient_DS, val_NoLiver_patient_DS = climain(data_path=self.datafolder,
                                                                                 Input_worker=self.worker,
                                                                                 mode='val',
                                                                                 dataset_mode=self.mode,
                                                                                 clean=self.clean
                                                                                 )

        # val_ds is always 4 patients
        # todo: all fully  1 2
        if self.mode == 1 or self.mode == 2 or self.mode == 6:
            self.train_ds = train_ds_alllabel
            self.val_ds = val_ds_alllabel
        # todo: only leaky 7
        elif self.mode == 7 or self.mode == 8:
            self.val_ds = val_Nolung_patient_DS + val_NoLiver_patient_DS
            self.train_ds = train_Nolung_patient_DS + train_NoLiver_patient_DS
        # todo: fully+ leaky 3 4
        else:
            self.val_ds = val_Nolung_patient_DS + val_NoLiver_patient_DS
            self.train_ds = train_ds_alllabel + train_Nolung_patient_DS + train_NoLiver_patient_DS

    def train_dataloader(self, cache=None):

        train_loader = DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.worker,
            pin_memory=torch.cuda.is_available(),
            collate_fn=list_data_collate

        )
        return train_loader

    def val_dataloader(self, cache=None):

        val_loader = DataLoader(
            dataset=self.val_ds,
            batch_size=8,
            num_workers=self.worker,
            pin_memory=torch.cuda.is_available(),
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
