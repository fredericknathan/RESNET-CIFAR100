import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from torch.autograd import Variable
import sys
import numpy as np

# WIDERESNET v2

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=10, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

# WIDERESNET v1

# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
#         super(BasicBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.droprate = dropRate
#         self.equalInOut = (in_planes == out_planes)
#         self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
#                                padding=0, bias=False) or None
#     def forward(self, x):
#         if not self.equalInOut:
#             x = self.relu1(self.bn1(x))
#         else:
#             out = self.relu1(self.bn1(x))
#         out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, training=self.training)
#         out = self.conv2(out)
#         return torch.add(x if self.equalInOut else self.convShortcut(x), out)

# class NetworkBlock(nn.Module):
#     def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
#         super(NetworkBlock, self).__init__()
#         self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
#     def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
#         layers = []
#         for i in range(int(nb_layers)):
#             layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
#         return nn.Sequential(*layers)
#     def forward(self, x):
#         return self.layer(x)

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes, depth=28, widen_factor=10, dropRate=0.3):
#         super(SimpleCNN, self).__init__()
#         nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
#         assert((depth - 4) % 6 == 0)
#         n = (depth - 4) / 6
#         block = BasicBlock
#         # 1st conv before any network block
#         self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         # 1st block
#         self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
#         # 2nd block
#         self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
#         # 3rd block
#         self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
#         # global average pooling and classifier
#         self.bn1 = nn.BatchNorm2d(nChannels[3])
#         self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(nChannels[3], num_classes)
#         self.nChannels = nChannels[3]

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         out = out.view(-1, self.nChannels)
#         return self.fc(out)

# """
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SimpleCNN, self).__init__()
#         # Convolutional layers: progressively increase number of filters (3 -> 32 -> 64 -> 128)
#         # 3x3 kernels with padding=1 maintain spatial dimensions before pooling
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling reduces spatial dimensions by half
#         # Fully connected layers: flatten feature maps and classify
#         self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 128 channels * 4x4 spatial resolution
#         self.fc2 = nn.Linear(512, num_classes)
#         self.dropout = nn.Dropout(0.5)  # Dropout for regularization

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 4 * 4)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x
# """

def create_model(num_classes, device):
    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)
    return model
