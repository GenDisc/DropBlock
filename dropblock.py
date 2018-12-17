'''
Resnet implementation code from 
"https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py"

'''
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DropBlock(nn.Module):
    def __init__(self, blocksize, keep_prob=1.0):
        super(DropBlock, self).__init__()
        self.blocksize = blocksize
        self.keep_prob = keep_prob

    def forward(self, x):
        if not self.training:
            return x
        else:
            feat_size_w, feat_size_h = x.size(2), x.size(3)
            residual = self.blocksize // 2
            
            # gamma : (1-keep) = feat^2 : (feat-block+1)^2 * block^2
            gamma = (1 - self.keep_prob)*(feat_size_h*feat_size_w) \
            / (self.blocksize**2*(feat_size_h - self.blocksize + 1) \
            *(feat_size_w - self.blocksize + 1))
            
            mask = torch.ones(x.size(0), x.size(1), x.size(2) - residual*2, x.size(3) - residual*2) * gamma
            mask = torch.bernoulli(mask)
            mask = mask.to(x.device)
            mask = F.pad(mask, (residual, residual, residual, residual), value=0)
            mask = F.conv2d(mask, torch.ones((x.size(1), 1, self.blocksize, self.blocksize)).to(x.device),
                    groups=x.size(1), padding=residual)
            mask.clamp_(max=1)
            mask = 1 - mask
            normalize = mask.numel()/mask.sum()

            return x * mask * normalize
    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False, dropblock=False):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.dropblock_flag = dropblock
        if self.dropblock_flag:
            self.dropblock = DropBlock(5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def keepprob_update(self, prob):
        assert self.dropblock_flag==True
        self.dropblock.keep_prob = prob
        print('Keep prob : %f'%self.dropblock.keep_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        if self.dropblock_flag:
            x = self.dropblock(x)
        x = self.layer2(x)
        if self.dropblock_flag:
            x = self.dropblock(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet56(**kwargs):
    model = ResNet(BasicBlock, [9, 9, 9], **kwargs)
    return model