# Author : Bryce Xu
# Time : 2019/11/18
# Function: 卷积神经网络

import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Network(nn.Module):
    def __init__(self, in_dim=1, hid_dim=64, out_dim=64):
        super(Network, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(in_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, out_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
