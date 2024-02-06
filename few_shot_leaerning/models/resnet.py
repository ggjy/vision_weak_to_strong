import torch.nn as nn

from .models import register


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):
    
    def __init__(self, inplanes, planes, downsample=None, maxpool=None):
        super().__init__()
        
        self.relu = nn.LeakyReLU(0.1)
        
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        
        self.downsample = downsample
        
        # self.maxpool = nn.MaxPool2d(2)
        self.maxpool = maxpool
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        identity = self.downsample(x) if self.downsample else x
        
        out += identity
        out = self.relu(out)
        
        out = self.maxpool(out) if self.maxpool else out
        
        return out


class ResNet12(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.inplanes = 3
        
        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])
        
        self.out_dim = channels[3]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, planes):
        downsample = nn.Sequential(
                conv1x1(self.inplanes, planes),
                norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x


class ResNet(nn.Module):
    
    def __init__(self, layers, channels):
        super().__init__()
        
        self.inplanes = 3
        
        self.layer1 = self._make_layer(layers[0], channels[0])
        self.layer2 = self._make_layer(layers[1], channels[1])
        self.layer3 = self._make_layer(layers[2], channels[2])
        self.layer4 = self._make_layer(layers[3], channels[3])
        
        self.out_dim = channels[3]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, blocks, planes):
        downsample = nn.Sequential(
                conv1x1(self.inplanes, planes),
                norm_layer(planes),
        )
        maxpool = nn.MaxPool2d(2)
        
        layers = [Block(self.inplanes, planes, downsample, maxpool)]
        
        for _ in range(1, blocks):
            layers.append(Block(planes, planes))
        
        block = nn.Sequential(*layers)
        
        self.inplanes = planes
        return block
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x


@register('resnet12')
def resnet12():
    return ResNet12([64, 128, 256, 512])


@register('resnet12_bottle')
def resnet12_bottle():
    return ResNet([1, 1, 1, 1], [64, 128, 256, 512])


@register('resnet18_bottle')
def resnet18_bottle():
    return ResNet([1, 2, 2, 1], [64, 128, 256, 512])


@register('resnet36_bottle')
def resnet36_bottle():
    return ResNet([2, 3, 5, 2], [64, 128, 256, 512])


@register('resnet50_bottle')
def resnet50_bottle():
    return ResNet([3, 4, 6, 3], [64, 128, 256, 512])


@register('resnet12-wide')
def resnet12_wide():
    return ResNet12([64, 160, 320, 640])
