import torch
import torchmetrics.functional as f

c=torch.randint(0,10,(3,4)).float()
print(c)
avg_precision = torch.mean(torch.stack([x for x in c]),dim=0)
print(avg_precision)

print(c.size())

