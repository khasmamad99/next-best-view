from typing import Union

import torch.nn as nn


class Conv3DBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: Union[int, str] = 0
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn   = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Basic3DCNN(nn.Module):

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv_block_1 = Conv3DBlock(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.conv_block_2 = Conv3DBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv_block_3 = Conv3DBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.avg_pooling  = nn.AdaptiveAvgPool3d([1, 1, 1])
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.avg_pooling(x)
        num_batches = x.shape[0]
        x = x.view(num_batches, -1)
        x = self.fc(x)
        return x


