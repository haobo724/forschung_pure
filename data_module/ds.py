"""

  MODE 1: ALL labeled patients --->30
  MODE 2: ALL labeled patients --->10
  MODE 3: ALL labeled patients --->10 + NoLiver ---10 + NoLung ---10 --->30
  MODE 4: ALL labeled patients --->10 + NoLiver ---10 + NoLung ---10 --->30 with modifiy label/trainstep

+--------+--------------+-----------------------+----------------------+-----+----------+-------+----------------+
|        | Full Labeled | Labeled without Liver | Labeled without lung | Sum | Modified | Epoch | Misslabel Rate |
+--------+--------------+-----------------------+----------------------+-----+----------+-------+----------------+
| Mode 1 |      18      |           0           |           0          |  18 |   False  | 30    |       0%       |
+--------+--------------+-----------------------+----------------------+-----+----------+-------+----------------+
| Mode 2 |       9      |           0           |           0          |  6  |   False  | 15    |       0%       |
+--------+--------------+-----------------------+----------------------+-----+----------+-------+----------------+
| Mode 3 |       6      |           6           |           6          |  12 |   False  | 30    |       33%      |
+--------+--------------+-----------------------+----------------------+-----+----------+-------+----------------+
| Mode 4 |       6      |           6           |           6          |  18 |   True   | 30    |       33%      |
+--------+--------------+-----------------------+----------------------+-----+----------+-------+----------------+
| Mode 5（7） |       0      |           6           |           6          |  12 |   False  | 30    |       50%      |
+--------+--------------+-----------------------+----------------------+-----+----------+-------+----------------+




    """

import os
import matplotlib.pyplot as plt
import monai
import numpy as np
import re
import monai.transforms as mxform
import data_module.custom_transform as custom_transform
import logging
from monai.data.utils import list_data_collate

def leakylabel_generator(img_list, mask_list, leakylabellist,root_str):
    leakylabel_img = []
    leakylabel_mask = []
    for index in range(len(img_list)):
        for leakylabel in leakylabellist:

            if re.findall(leakylabel, img_list[index]):
                leakylabel_img.append(img_list[index])
                leakylabel_mask.append(mask_list[index])
    noliver_img = [root_str + '/' + i for i in leakylabel_img]
    noliver_mask = [root_str + '/' + i for i in leakylabel_mask]
    return noliver_img, noliver_mask

def getdataset(data_path):#mask 和 img在同一目录
    img_list = []
    mask_list = []
    root_str=''
    for index, (root, dir, files) in enumerate(os.walk(data_path)):
        for file in files:
            if file.endswith('_ct.nii.gz'):
                img_list.append(file)
            if file.endswith('_seg.nii.gz'):
                root_str = root
                mask_list.append(file)

    img_list = sorted(img_list)
    mask_list = sorted(mask_list)
    return root_str,img_list,mask_list

def getpatient_name(img_list):
    patient_name = []
    patient_num=0
    curName = ''
    for img_str in img_list:
        pattern_index = img_str.find('_', 0)
        pattern = img_str[:pattern_index]
        if curName == pattern:
            continue
        else:
            curName = pattern
            patient_name.append(curName)
            patient_num += 1
    return patient_name,patient_num

def get_xform(mode="train", keys=("image", "label","leaky"),leaky=None,leakylist=None):
    xforms = [
        mxform.LoadImaged(keys[:2]),
        custom_transform.Transposed(keys[:2], (1, 0)),
        mxform.AddChanneld(keys[:2]),
        custom_transform.NormalizeLabeld(keys=['label'], from_list=[0,7, 8, 6], to_list=[0,1, 2, 3]),
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
        # xforms = xforms[:-2]
        dtype = (np.float32,np.float32)

    if leaky =='liver':
        xforms.extend([
            custom_transform.Leakylabel(keys=["leaky"], leakylist=leakylist,leaky=leaky),
            custom_transform.NormalizeLabeld(keys=['label'], from_list=[0,1, 2, 3], to_list=[0, 0, 2, 3]),

        ])
    elif leaky == 'lung':
        xforms.extend([
            custom_transform.Leakylabel(keys=["leaky"], leakylist=leakylist,leaky=leaky),
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

def Full_Return(Fulllabel_str_list_T,Nolung_str_list,NoLiver_str_list,
                    Fulllabel_str_list_mask_T,Nolung_str_list_mask,NoLiver_str_list_mask,
                    NoLung_name,NoLiver_name,mode):

    # TODO: 转化为dict
    keys = ("image", "label", "leaky")
    Alllabel_patient = [
        {keys[0]: img, keys[1]: seg, keys[2]: seg} for img, seg in
        zip(Fulllabel_str_list_T, Fulllabel_str_list_mask_T)
    ]
    Nolung_patient = [
        {keys[0]: img, keys[1]: seg, keys[2]: seg} for img, seg in
        zip(Nolung_str_list, Nolung_str_list_mask)
    ]

    NoLiver_patient = [
        {keys[0]: img, keys[1]: seg, keys[2]: seg} for img, seg in
        zip(NoLiver_str_list, NoLiver_str_list_mask)
    ]
    # Todo：三种transform
    train_transform__Nolung = get_xform(mode=mode, leaky='lung', leakylist=NoLung_name)
    train_transform__NoLiver = get_xform(mode=mode, leaky='liver', leakylist=NoLiver_name)
    train_transform__Alllabel = get_xform(mode=mode, leaky='all')

    # Todo：对应三种Ds

    train_Nolung_patient_DS = monai.data.CacheDataset(
        data=Nolung_patient,
        transform=train_transform__Nolung,
        num_workers=4

    )
    train_NoLiver_patient_DS = monai.data.CacheDataset(
        data=NoLiver_patient,
        transform=train_transform__NoLiver,
        num_workers=4

    )

    train_ALLlabel_patient_DS = monai.data.CacheDataset(
        data=Alllabel_patient,
        transform=train_transform__Alllabel,
        num_workers=4


    )
    return train_ALLlabel_patient_DS, train_Nolung_patient_DS, train_NoLiver_patient_DS




def Part_Return(Nolung_str_list,NoLiver_str_list,
                    Nolung_str_list_mask,NoLiver_str_list_mask,
                    NoLung_name,NoLiver_name,mode):
    # TODO: 转化为dict
    keys = ("image", "label", "leaky")

    Nolung_patient = [
        {keys[0]: img, keys[1]: seg, keys[2]: seg} for img, seg in
        zip(Nolung_str_list, Nolung_str_list_mask)
    ]

    NoLiver_patient = [
        {keys[0]: img, keys[1]: seg, keys[2]: seg} for img, seg in
        zip(NoLiver_str_list, NoLiver_str_list_mask)
    ]
    # Todo：三种transform
    train_transform__Nolung = get_xform(mode=mode, leaky='lung', leakylist=NoLung_name)
    train_transform__NoLiver = get_xform(mode=mode, leaky='liver', leakylist=NoLiver_name)

    # Todo：对应三种Ds

    train_Nolung_patient_DS = monai.data.CacheDataset(
        data=Nolung_patient,
        transform=train_transform__Nolung,
        num_workers=4

    )
    train_NoLiver_patient_DS = monai.data.CacheDataset(
        data=NoLiver_patient,
        transform=train_transform__NoLiver,
        num_workers=4

    )

    return  [],train_Nolung_patient_DS, train_NoLiver_patient_DS


def climain(data_path=r'F:\Forschung\multiorganseg\data\train_2D',Input_worker=4,mode='train',dataset_mode=6):
    data_path=data_path
    root_str,img_list,mask_list = getdataset(data_path)

    patient_name,patient_num=getpatient_name(img_list)
    patient_name=sorted(patient_name)
    if dataset_mode ==7:
        NoLiver_name = patient_name[10:18]  # 10
        NoLiver_name_V = patient_name[18:20]  # 10
        NoLung_name = patient_name[20:28]  # 10
        NoLung_name_V = patient_name[28:30]  # 10
        # TODO: 根据病人名字分割出来三组list，保存的是路径

        Nolung_str_list, Nolung_str_list_mask = leakylabel_generator(img_list, mask_list,
                                                                     NoLung_name, root_str)
        NoLiver_str_list, NoLiver_str_list_mask = leakylabel_generator(img_list, mask_list,
                                                                       NoLiver_name, root_str)

        Nolung_str_list_V, Nolung_str_list_mask_V = leakylabel_generator(img_list, mask_list,
                                                                         NoLung_name_V, root_str)
        NoLiver_str_list_V, NoLiver_str_list_mask_V = leakylabel_generator(img_list, mask_list,
                                                                           NoLiver_name_V, root_str)
        if mode == 'train':

            return Part_Return( Nolung_str_list, NoLiver_str_list,
                                      Nolung_str_list_mask, NoLiver_str_list_mask,
                                     NoLung_name, NoLiver_name,mode=mode )
        else:

            return Part_Return( Nolung_str_list_V, NoLiver_str_list_V,
                                    Nolung_str_list_mask_V, NoLiver_str_list_mask_V,
                                   NoLung_name_V, NoLiver_name_V, mode=mode )

    if dataset_mode ==5:
        print(f'[INFO] TEST Dataset_mode: {dataset_mode}')
        fulllabeled_name_sub_T=patient_name[32:34]
        if len(fulllabeled_name_sub_T)==1:
            fulllabeled_name_sub_T=[fulllabeled_name_sub_T]
        Fulllabel_str_list_T, Fulllabel_str_list_mask_T = leakylabel_generator(img_list, mask_list, fulllabeled_name_sub_T,
                                                                               root_str)
        keys = ("image", "label", "leaky")
        num_alllabel = len(Fulllabel_str_list_T)
        test_patient = [
            {keys[0]: img, keys[1]: seg, keys[2]: seg} for img, seg in
            # zip(Fulllabel_str_list_T[350:360]+Fulllabel_str_list_T[150:155], Fulllabel_str_list_mask_T[350:360]+Fulllabel_str_list_mask_T[150:155])
            zip(Fulllabel_str_list_T, Fulllabel_str_list_mask_T)
        ]
        test_ALLlabel_patient_DS = monai.data.CacheDataset(
            data=test_patient,
            transform=get_xform(mode='test', leaky='all'),
            num_workers=Input_worker

        )
        return test_ALLlabel_patient_DS, [], []

    if dataset_mode ==6:
        print(f'[INFO] New Dataset_mode: {dataset_mode}')
        print(f'TEST CacheDataset')
        fulllabeled_name_sub_T =[ patient_name[0]]  # 前5个
        fulllabeled_name_sub_V = [patient_name[2]] # 前5个

        Fulllabel_str_list_T, Fulllabel_str_list_mask_T = leakylabel_generator(img_list, mask_list,
                                                                           fulllabeled_name_sub_T, root_str)

        Fulllabel_str_list_V, Fulllabel_str_list_mask_V = leakylabel_generator(img_list, mask_list,
                                                                               fulllabeled_name_sub_V, root_str)

        keys = ("image", "label", "leaky")

        if mode == 'train':
            Alllabel_patient_train = [
                {keys[0]: img, keys[1]: seg, keys[2]: seg} for img, seg in
                zip(Fulllabel_str_list_T[:10], Fulllabel_str_list_mask_T[:10])
            ]
            train_ALLlabel_patient_DS = monai.data.CacheDataset(
                data=Alllabel_patient_train,
                transform=get_xform(mode=mode, leaky='all'),
                num_workers=Input_worker
            )
            return train_ALLlabel_patient_DS, [], []
        else:
            Alllabel_patient_val = [
                {keys[0]: img, keys[1]: seg, keys[2]: seg} for img, seg in
                zip(Fulllabel_str_list_V, Fulllabel_str_list_mask_V)
            ]
            val_ALLlabel_patient_DS = monai.data.CacheDataset(
                data=Alllabel_patient_val,
                transform=get_xform(mode=mode, leaky='all'),
                num_workers=Input_worker

            )
            return val_ALLlabel_patient_DS, [], []

    if dataset_mode ==1 or dataset_mode==2:
        print(f'[INFO] Dataset_mode: {dataset_mode}')
        print(f'[INFO] ALL labeled patients --->30') if dataset_mode == 1 else logging.info(f'[INFO] ALL labeled patients --->10')


        fulllabeled_name_sub_T=patient_name[:24] if dataset_mode ==1 else patient_name[:8]
        fulllabeled_name_sub_V=patient_name[24:30] if dataset_mode ==1 else patient_name[8:10]
        

        Fulllabel_str_list_T, Fulllabel_str_list_mask_T = leakylabel_generator(img_list, mask_list, fulllabeled_name_sub_T,
                                                                           root_str)
        Fulllabel_str_list_V, Fulllabel_str_list_mask_V = leakylabel_generator(img_list, mask_list, fulllabeled_name_sub_V,
                                                                           root_str)
        keys = ("image", "label","leaky")
        num_alllabel = len(Fulllabel_str_list_T)

        if mode == 'train':
            Alllabel_patient_train = [
                {keys[0]: img, keys[1]: seg, keys[2]: seg} for img, seg in
                zip(Fulllabel_str_list_T, Fulllabel_str_list_mask_T)
            ]
            train_ALLlabel_patient_DS = monai.data.CacheDataset(
                data=Alllabel_patient_train,
                transform=get_xform(mode=mode, leaky='all'),
                num_workers=Input_worker

            )
            return train_ALLlabel_patient_DS,[],[]
        else:
            Alllabel_patient_val = [
                {keys[0]: img, keys[1]: seg, keys[2]: seg} for img, seg in
                zip(Fulllabel_str_list_V, Fulllabel_str_list_mask_V)
            ]
            val_ALLlabel_patient_DS = monai.data.CacheDataset(
                data=Alllabel_patient_val,
                transform=get_xform(mode=mode, leaky='all'),
                num_workers=Input_worker

            )
            return val_ALLlabel_patient_DS,[],[]

    if dataset_mode ==3 or dataset_mode==4:
        print(f'[INFO] Dataset_mode: {dataset_mode}')
        print(f'[INFO] MODE 3: ALL labeled patients ---10 + NoLiver ---10 + NoLung ---10 --->30') if dataset_mode == 3 else print(
            f'  [INFO] MODE 4: ALL labeled patients ---10 + NoLiver ---10 + NoLung ---10 --->30 with modifiy label/trainstep')

        fulllabeled_name_sub_T = patient_name[:8]  # 前十个
        fulllabeled_name_sub_V = patient_name[8:10]  # 前十个


        NoLiver_name=patient_name[10:18]#10
        NoLiver_name_V=patient_name[18:20]#10

        NoLung_name=patient_name[20:28]#10
        NoLung_name_V=patient_name[28:30]#10
        #TODO: 根据病人名字分割出来三组list，保存的是路径

        Fulllabel_str_list_T, Fulllabel_str_list_mask_T=leakylabel_generator(img_list, mask_list,
                                                                         fulllabeled_name_sub_T, root_str)
        Nolung_str_list, Nolung_str_list_mask=leakylabel_generator(img_list, mask_list,
                                                                         NoLung_name, root_str)
        NoLiver_str_list, NoLiver_str_list_mask=leakylabel_generator(img_list, mask_list,
                                                                     NoLiver_name, root_str)

        Fulllabel_str_list_V, Fulllabel_str_list_mask_V = leakylabel_generator(img_list, mask_list,
                                                                               fulllabeled_name_sub_V,
                                                                               root_str)
        Nolung_str_list_V, Nolung_str_list_mask_V = leakylabel_generator(img_list, mask_list,
                                                                     NoLung_name_V, root_str)
        NoLiver_str_list_V, NoLiver_str_list_mask_V = leakylabel_generator(img_list, mask_list,
                                                                       NoLiver_name_V, root_str)



        if mode=='train':

            return Full_Return(Fulllabel_str_list_T,Nolung_str_list,NoLiver_str_list,
                        Fulllabel_str_list_mask_T,Nolung_str_list_mask,NoLiver_str_list_mask,
                                       NoLung_name,NoLiver_name,mode=mode )
        else:

            return Full_Return(Fulllabel_str_list_V, Nolung_str_list_V, NoLiver_str_list_V,
                          Fulllabel_str_list_mask_V, Nolung_str_list_mask_V, NoLiver_str_list_mask_V,
                          NoLung_name_V, NoLiver_name_V,mode=mode  )


def test():
    train_ds_alllabel, train_Nolung_patient_DS, train_NoLiver_patient_DS=climain(mode='train',dataset_mode=7)
    train_ds = train_Nolung_patient_DS + train_NoLiver_patient_DS
    bs=4
    train_loader = monai.data.DataLoader(
        train_ds,
        shuffle=True,
        batch_size=bs,
        num_workers=4,
        pin_memory=True,
        collate_fn=list_data_collate

    )

    for idx,data in enumerate(train_loader):
        x=data["image"]
        y=data["label"]
        z=data["leaky"]

        fig, axs = plt.subplots(2, bs)
        for i in range(bs):
            axs[0,i].imshow(y[i,0,...])
            axs[0,i].set_title(f'{z[i]}')
            axs[1,i].imshow(x[i,0,...])
            axs[1,i].set_title(f'{z[i]}')
        plt.show()
if __name__ == "__main__":
    test()
