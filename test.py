import numpy as np
import torch
import pickle
import torchmetrics.functional as f
import matplotlib.pyplot as plt
from collections import Counter


a=torch.ones((512,512))
a[0,2]=10
a[4,2]=10
# t= torch.tensor([10])
x=torch.where(a==10)[0]
y=torch.where(a==10)[1]

for i,j in zip (x,y):
    print(i,j)
    a[i,j]=304


# print(a[cc[0],cc[1]])
print(a[0,2])
print(a[4,2])

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



