import numpy as np
import torch
import torchmetrics.functional as f

c=np.array([1,1,2,2,3])
print(c)
cord_zusatz = np.argwhere(c== 2 )
print(cord_zusatz)


