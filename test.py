import time

import numpy as np
import torch
from measures import calculate_eval_matrix, calculate_union, calculate_intersection

import cv2
import torchmetrics.functional as f
import matplotlib.pyplot as plt
import pycuda
from collections import Counter

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
a=np.ones((400,400))
a[0,2]=10
a[2,2]=10

b=np.ones((400,400))
b[0,2]=0
c=np.ones((400,400))

time_mark1=time.time()
cord1=a==10
cord2=b==0
cord3=np.bitwise_and(cord1,cord2)

c[cord3==True]=6
time_mark2=time.time()
print(time_mark2-time_mark1)

cord1_2=np.where(a == 10)
cord_y = cord1_2[0]
cord_x = cord1_2[1]
real_tmp = b[cord_y, cord_x] == 0
idxx = np.where(real_tmp == True)
realcord_y = cord_y[idxx]
realcord_x = cord_x[idxx]

# if torch.max(y_copy[idx,...])!=0:
c[realcord_y, realcord_x] = 6
time_mark3=time.time()
print(time_mark3-time_mark2)



i=np.where(c==6)
print(i)

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
