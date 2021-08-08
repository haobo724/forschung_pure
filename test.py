import numpy as np
import torch
import pickle
import torchmetrics.functional as f
import matplotlib.pyplot as plt
import pycuda
from collections import Counter
a=torch.zeros((4,3,110,200))
#设定某个点的三通道都是10，(假设每张图都一样）
a[:,:,2,4]=10
#提取一个通道，找出等于10的xy坐标
cord=torch.where(a[:,0,...]==10)
#根据上边的坐标把它改成50
a[:,0,cord[1],cord[2]]=50
print(len(cord))
cord=torch.where(a==50)
print(cord[0],cord[1],cord[2],cord[3])


# a=torch.ones((512,512))
# a[0,2]=10
# a[4,2]=10
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



