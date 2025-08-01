import torch.nn as nn
import torch.nn.functional as F

class WideBasic(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)

class WRN32x8(nn.Module):
    def __init__(self, num_classes=100, k=8):
        super().__init__()
        n = 5
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(16*k, n, 1)
        self.layer2 = self._make_layer(32*k, n, 2)
        self.layer3 = self._make_layer(64*k, n, 2)
        self.bn = nn.BatchNorm2d(64*k)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64*k, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, out_planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(WideBasic(self.in_planes, out_planes, s))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn(x))
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc(x)
