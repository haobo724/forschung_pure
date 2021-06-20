import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    "...-> conv -> BN -> relu -> conv -> BN -> relu ..."
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)




class DownBlock(nn.Module):
    """...conv_block->max_pooling(2)..."""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        # the skip should be before maxpooling
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.down_block = nn.Sequential(
        #     ConvBlock(in_channels, out_channels),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        self.skip_tensor = None

    def forward(self, x):
        x = self.conv_block(x)
        self.skip_tensor = x
        return self.pooling(x)




class UpBlock(nn.Module):
    """...Upsampling(2)->concatenation->conv_block"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, y):
        # print(x.shape)
        x = self.up_conv(x)

        x_left = (y.size()[3] - x.size()[3]) // 2
        x_right = y.size()[3] - x.size()[3] - x_left
        y_left = (y.size()[2] - x.size()[2]) // 2
        y_right = y.size()[2] - x.size()[2] - y_left

        x = torch.nn.functional.pad(x, [y_left, y_right, x_left, x_right])
        x = torch.cat([x, y], dim=1)

        return self.conv_block(x)




class BasicUnet(nn.Module):

    def __init__(self, **kwargs):
        # some instructions here
        super().__init__()

        self.config_kwarg = kwargs
        self.depth = kwargs.get('depth', 4)
        self.nfilters = kwargs.get('nfilters', 32)
        self.in_channels = kwargs.get('in_channels', 1)
        self.out_channels = kwargs.get('out_channels', 4)

        # encoder path
        self.encoder_path = nn.ModuleList()
        self.encoder_path.append(DownBlock(self.in_channels, self.nfilters))
        pre_nfilters = self.nfilters
        for i in range(self.depth - 2):
            self.encoder_path.append(
                DownBlock(pre_nfilters, 2 * pre_nfilters)
            )
            pre_nfilters = 2 * pre_nfilters

        # bottom
        self.bottom_block = ConvBlock(pre_nfilters, 2*pre_nfilters)
        pre_nfilters = 2 * pre_nfilters

        # decoder path
        self.decoder_path = nn.ModuleList()
        for i in range(self.depth - 1):
            self.decoder_path.append(
                UpBlock(pre_nfilters, pre_nfilters // 2)
            )
            pre_nfilters = pre_nfilters // 2
        self.out_conv = nn.Conv2d(pre_nfilters, self.out_channels, kernel_size=1)
        self.init_weights()
    def forward(self, x):
        cache = []
        for down in self.encoder_path:
            x = down(x)
            cache.append(down.skip_tensor)

        x = self.bottom_block(x)

        for up in self.decoder_path:
            x = up(x, cache.pop())

        return self.out_conv(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def export_config(self):
        return self.config_kwarg




if __name__ == '__main__':
    config = {
        'in_channels': 1,
        'out_channels': 4,
        'depth': 4,
        'nfilters': 32
    }
    # model = BasicUnet(**config)
    model = BasicUnet(**config)
    # print(model)
    print(model.export_config())

    # model2 = BasicUnet(**model.export_config())
    # print(model2)

    x = torch.rand(1, 1, 128, 128)
    y = model(x)
    print(y.shape)