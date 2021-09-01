"""
FileName:	Model_DSRF3D_v2.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-08-04 18:53:45
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSRF3D_v2(nn.Module):
    def __init__(self, num_classes=10, keepfc=True):
        super(DSRF3D_v2, self).__init__()
        # dimensions of the 3D image. Channels, Depth, Height, Width
        C = 1
        D = 24
        H = 24
        W = 24
        self.keepfc = keepfc

        self.conv1 = nn.Conv3d(C, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if keepfc:
            self.fc = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        # 1 x 24 x 24 x 24 -> 32 x 24 x 24 x 24
        x = F.relu(self.conv1(x))
        # 32 x 24 x 24 x 24 -> 32 x 24 x 24 x 24
        x = F.relu(self.conv2(x))
        # 32 x 24 x 24 x 24 -> 32 x 12 x 12 x 12
        x = F.max_pool3d(x, kernel_size=2, stride=2, padding=0)
        # 32 x 12 x 12 x 12 -> 64 x 12 x 12 x 12
        x = F.relu(self.conv3(x))
        # 64 x 12 x 12 x 12 -> 64 x 12 x 12 x 12
        x = F.relu(self.conv4(x))
        # 64 x 12 x 12 x 12 -> 64 x 6 x 6 x 6
        x = F.max_pool3d(x, kernel_size=2, stride=2, padding=0)
        # 64 x 6 x 6 x 6 -> 128 x 6 x 6 x 6
        x = F.relu(self.conv5(x))
        # 128 x 6 x 6 x 6 -> 128 x 6 x 6 x 6
        x = F.relu(self.conv6(x))
        # 128 x 6 x 6 x 6 -> 128 x 1 x 1 x 1
        x = self.avgpool(x)
        # 128 x 1 x 1 x 1 -> 128
        x = torch.flatten(x, 1)
        if self.keepfc:
            # 128 -> num_classes
            x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DSRF3D_v2(num_classes=10)
    print(model)
    print(list(model.modules()))
