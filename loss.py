import torch
import numpy as np
import monai

class dice_loss_(torch.nn.Module):
    def __init__(self, smooth=0.1):
        super(dice_loss_, self).__init__()
        self.smooth = smooth

    def __call__(self, y_hat, y):

        bs, out_channel, w, h = y_hat.shape

        y_hat = torch.sigmoid(y_hat)
        y = y.unsqueeze(1)
        y_onehot = torch.zeros_like(y_hat)
        y_onehot.scatter_(1, y.type(torch.int64), 1)

        y_flat = y_onehot.view(-1)
        y_hat_flat = y_hat.view(-1)

        intersection = (y_flat * y_hat_flat).sum()

        return 1 - (2 * intersection + self.smooth) / (y_flat.sum() + y_hat_flat.sum() + self.smooth)


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth = 0.1, weight=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

        if weight is None:
            self.weights = 1
        else:
            self.weights = torch.Tensor(weight)

    def __call__(self, y_hat, y):
        # calculate dice loss using torch methods
        # y_hat.shape = [bs, out_channel, w, h]
        # y.shape = [bs, w, h]

        # transform y to one-hot y
        bs, out_channel, w, h = y_hat.shape
        y_hat = torch.sigmoid(y_hat)

        y = y.unsqueeze(1)
        y_onehot = torch.zeros_like(y_hat)
        y_onehot.scatter_(1, y.type(torch.int64), 1)

        # set up weights

        accumulation = torch.zeros(bs, device=y.device)
        for i in range(out_channel):
            # Go through all channels and calculate the loss

            # Extract the i-th channel and flatten
            y_hat_flat = y_hat[:, i, :, :].contiguous().view(bs, -1)
            y_onehot_flat = y_onehot[:, i, :, :].contiguous().view(bs, -1)
            intersection = torch.sum(y_hat_flat * y_onehot_flat, dim=1)

            A_sum = torch.sum(y_hat_flat, dim=1)
            B_sum = torch.sum(y_onehot_flat, dim=1)

            l = 1 - ((2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth))

            if type(self.weights) is int:
                accumulation += l
            else:
                accumulation += l * self.weights[i]

        if type(self.weights) is int:
            accumulation = torch.sum(accumulation) / (bs * out_channel)
        else:
            accumulation = torch.sum(accumulation) / (torch.sum(self.weights) * bs)
        return accumulation

class DiceCELoss(torch.nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self, weight, dice=True):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True)
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=torch.tensor(np.array(weight)).float().cuda())

    def forward(self, y_pred, y_true):
        # y_pred = torch.sigmoid(y_pred)
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy

class CELoss(torch.nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self, weight):
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=torch.tensor(np.array(weight)).float().cuda())

    def forward(self, y_pred, y_true):
        # y_pred = torch.sigmoid(y_pred)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return cross_entropy