import torch
import torchmetrics.functional as f

x = torch.tensor([1, 2, 3])
y = torch.tensor([1, 2, 3])
c=((x!=4)*(y!=4)).sum()/len(y)
print(c)





# pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
#                             [0.05, 0.85, 0.05, 0.05],
#                              [0.05, 0.05, 0.85, 0.05],
#                               [0.05, 0.05, 0.05, 0.85]])

pred = torch.tensor([         [0, 1, 2, 0, 3],
                              [0, 2, 1, 3, 3],
                              [0, 3, 2, 1, 3],
                              [0, 3, 1, 2, 3]])
# pred = torch.tensor([[1, 1, 1, 1]]
#                     )
print(pred.size())
target = torch.tensor([       [0, 1, 2, 3, 3],
                              [0, 2, 1, 3, 3],
                              [0, 3, 2, 1, 3],
                              [0, 3, 1, 2, 3]])
s=f.dice_score(pred, target,reduction='none',bg=True)

print(s)
