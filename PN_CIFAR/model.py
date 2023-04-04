import torch.nn as nn

class BasicBlock(nn.Module):
  def __init__(self, in_features, out_features, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=(3,3), stride=(stride,stride), padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_features)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=(3,3), stride=(1,1), padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_features)
    self.relu2 = nn.ReLU(inplace=True)
    self.downsample = downsample

  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample:
      identity = self.downsample(x)
    out += identity
    out = self.relu2(out)
    return out


class ResNet12(nn.Module):
  def __init__(self):
    super(ResNet12, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu1 = nn.ReLU(inplace=True)
    self.in_channels = 64
    self.layer1 = self.make_layer(BasicBlock, 64, 1)
    self.layer2 = self.make_layer(BasicBlock, 128, 1, stride=2)
    self.layer3 = self.make_layer(BasicBlock, 256, 2, stride=2)
    self.layer4 = self.make_layer(BasicBlock, 512, 1, stride=2)

  def make_layer(self, block, out_channels, blocks, stride=1):
    layers = []
    downsample = None
    if stride != 1 or self.in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels, kernel_size=(1,1), stride=(stride,stride), bias=False),
            nn.BatchNorm2d(out_channels)
        )
    layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
    self.in_channels = out_channels
    for _ in range(1, blocks):
        layers.append(BasicBlock(self.in_channels, out_channels))
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = out.view(out.size(0), -1)
    return out