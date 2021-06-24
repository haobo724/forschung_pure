import numpy as np
import torch
import pickle
import torchmetrics.functional as f
import matplotlib.pyplot as plt
from collections import Counter



def load(path=None):
    print("loaded")
    with open("./lungrecord.pkl", 'rb') as f:
        tmp = pickle.load(f)
    return tmp
a=np.squeeze(load())
b=np.zeros_like(a)
c=np.zeros_like(a)

# cmap=
result = Counter(a)
print(result)
x=np.arange(a.shape[0])
plt.scatter( x,a,c=a.astype(np.uint8), s=1)
plt.colorbar()

plt.show()



