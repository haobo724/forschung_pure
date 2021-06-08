import numpy as np
import torch
import torchmetrics.functional as f

c=[2,3,4]
ct=torch.tensor(c)
print(c)
cord_zusatz = [1,2,3]
cord_zusatzt = torch.tensor(cord_zusatz)
cc= ct*cord_zusatzt
print(cc)


