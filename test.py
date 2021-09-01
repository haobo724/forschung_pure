import time

import numpy as np
import torch
# from torchvision import transforms as trans
from measures import calculate_eval_matrix, calculate_union, calculate_intersection
import monai
import cv2
import torchmetrics.functional as f
import matplotlib.pyplot as plt
import pycuda
from collections import Counter
import albumentations as trans

# pred=np.random.randint(0,4,(152,407,407))
# print(np.max(pred))
# print(np.min(pred))
# target=np.random.randint(0,4,(152,407,407))
#
# mat=calculate_eval_matrix(4,pred,pred)
# iou=calculate_intersection(mat)/calculate_union(mat)
# print(iou)

# from sklearn.preprocessing import LabelEncoder
# label=LabelEncoder()
# mask_path=r'F:\PixelAnnotationTool-master\PixelAnnotationTool\scripts_to_build\build\x64\PixelAnnotationTool_x64_v1.4.0-18-g4920f8e\images_test\road1_watershed_mask.png'
# mask=cv2.imread(mask_path,0)
# new_mask=label.fit_transform(mask.reshape(-1,1))
# label_detail=np.unique(new_mask)
# print(mask.shape)
# print(label_detail)
transform_demo = trans.Compose([
    # trans.ToPILImage(),
    trans.Resize(256, 256),
    # trans.ColorJitter(brightness=1,contrast=0,saturation=0.5,hue=0.5),
    trans.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255)
    trans.

])

img = cv2.imread(r'F:\Cam62-71\Cam62-71\20181215-00.11\img_00.11.31.jpg')
img_resize = cv2.resize(img, (256, 256)).astype(np.uint8)

print(img_resize.shape)
img_trans = transform_demo(image=img)['image'] * 255

print(img_trans.shape)

# img_trans=np.moveaxis(img_trans,0,2)
img_cat = np.concatenate((img_trans.astype(np.uint8), img_resize), axis=1)

cv2.imshow('hi', img_cat)
cv2.waitKey()

# c=monai.transforms.Compose(
#     [monai.transforms.LoadImaged('image'),
#      monai.transforms.CastToTyped('image',dtype=np.uint8),
#
#     monai.transforms.ToTensord('image')]
# )


# a = monai.transforms.LoadImaged(r(a, dtype=np.uint8)
# a= 'F:\Forschung\multiorganseg\data\train_2D\2300088WW0_448_seg.nii.gz')
# a = monai.transforms.CastToTyped

# a=monai.transforms.convert_to_numpy(a)
# d={'image':r'F:\Forschung\multiorganseg\data\train_2D\2300088WW0_448_seg.nii.gz'}
# result = c(d)
# img=result['image']
# print(img.size())
#
#
# x=[0,4,1,4]
# y=[2,2,4,5]
#
# idx=  (a[x,y]==10)
# i=torch.where(idx==True)
#
# print(i)


# t= torch.tensor([10])
# x = torch.where(a==10)[0]
# y = torch.where(a==10)[1]
#
# for i,j in zip (x,y):
#     print(i,j)
#     a[i,j]=304
#
#
# # print(a[cc[0],cc[1]])
# print(a[0,2])
# print(a[4,2])

# def load(path=None):
#     print("loaded")
#     with open("./lungrecord.pkl", 'rb') as f:
#         tmp = pickle.load(f)
#     return tmp
# a=np.squeeze(load())
# b=np.zeros_like(a)
# c=np.zeros_like(a)
#
# # cmap=
# result = Counter(a)
# print(result)
# x=np.arange(a.shape[0])
# plt.scatter( x,a,c=a.astype(np.uint8), s=1)
# plt.colorbar()
#
# plt.show()
