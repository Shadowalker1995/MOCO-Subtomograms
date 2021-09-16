"""
FileName:	Model_YOPO.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-09-15 23:30:10
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)
        self.elu = nn.ELU(alpha=1.0)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.99)
        # self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        # in_channels x ... -> out_channels x ...
        x = self.conv(x)
        # out_channels x ... -> out_channels x ...
        x = self.elu(x)
        # out_channels x ... -> out_channels x ...
        x = self.bn(x)
        return x


class YOPO(nn.Module):
    def __init__(self, num_classes=10, keepfc=True):
        super(YOPO, self).__init__()
        # dimensions of the 3D image. Channels, Depth, Height, Width
        C = 1
        D = 32
        H = 32
        W = 32
        # self.keepfc = keepfc

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # 1 x 32^3 -> 4 x 30^3
        self.conv_1_1 = BasicConv3d(1, 4, kernel_size=3, padding='valid')
        # 4 x 30^3 -> 5 x 28^3
        self.conv_1_2 = BasicConv3d(4, 5, kernel_size=3, padding='valid')
        # 5 x 28^3 -> 6 x 26^3
        self.conv_1_3 = BasicConv3d(5, 6, kernel_size=3, padding='valid')
        # 6 x 26^3 -> 7 x 24^3
        self.conv_1_4 = BasicConv3d(6, 7, kernel_size=3, padding='valid')
        # 7 x 24^3 -> 8 x 22^3
        self.conv_1_5 = BasicConv3d(7, 8, kernel_size=3, padding='valid')
        # 8 x 22^3 -> 9 x 20^3
        self.conv_1_6 = BasicConv3d(8, 9, kernel_size=3, padding='valid')

        # 1 x 32^3 -> 3 x 29^3
        self.conv_2_1 = BasicConv3d(1, 3, kernel_size=4, padding='valid')
        # 3 x 29^3 -> 4 x 26^3
        self.conv_2_2 = BasicConv3d(3, 4, kernel_size=4, padding='valid')
        # 4 x 26^3 -> 5 x 23^3
        self.conv_2_3 = BasicConv3d(4, 5, kernel_size=4, padding='valid')
        # 5 x 23^3 -> 6 x 20^3
        self.conv_2_4 = BasicConv3d(5, 6, kernel_size=4, padding='valid')

        # 1 x 32^3 -> 2 x 28^3
        self.conv_3_1 = BasicConv3d(1, 2, kernel_size=5, padding='valid')
        # 2 x 28^3 -> 3 x 24^3
        self.conv_3_2 = BasicConv3d(2, 3, kernel_size=5, padding='valid')
        # 3 x 24^3 -> 4 x 20^3
        self.conv_3_3 = BasicConv3d(3, 4, kernel_size=5, padding='valid')

        # 1 x 32^3 -> 1 x 26^3
        self.conv_4_1 = BasicConv3d(1, 1, kernel_size=7, padding='valid')
        # 1 x 26^3 -> 2 x 20^3
        self.conv_4_2 = BasicConv3d(1, 2, kernel_size=7, padding='valid')

        # (9+6+4+2)=21 x 20^3 -> 10 x 18^3
        self.conv_5_1 = BasicConv3d(21, 10, kernel_size=3, padding='valid')
        # 10 x 18^3 -> 11 x 16^3
        self.conv_5_2 = BasicConv3d(10, 11, kernel_size=3, padding='valid')
        # 11 x 16^3 -> 12 x 14^3
        self.conv_5_3 = BasicConv3d(11, 12, kernel_size=3, padding='valid')
        # 12 x 14^3 -> 13 x 12^3
        self.conv_5_4 = BasicConv3d(12, 13, kernel_size=3, padding='valid')
        # 13 x 12^3 -> 14 x 10^3
        self.conv_5_5 = BasicConv3d(13, 14, kernel_size=3, padding='valid')
        # 14 x 10^3 -> 15 x 8^3
        self.conv_5_6 = BasicConv3d(14, 15, kernel_size=3, padding='valid')
        # 15 x 8^3 -> 16 x 6^3
        self.conv_5_7 = BasicConv3d(15, 16, kernel_size=3, padding='valid')
        # 16 x 6^3 -> 17 x 4^3
        self.conv_5_8 = BasicConv3d(16, 17, kernel_size=3, padding='valid')
        # 17 x 4^3 -> 18 x 2^3
        self.conv_5_9 = BasicConv3d(17, 18, kernel_size=3, padding='valid')

        # (9+6+4+2)=21 x 20^3 -> 7 x 17^3
        self.conv_6_1 = BasicConv3d(21, 7, kernel_size=4, padding='valid')
        # 7 x 17^3 -> 8 x 14^3
        self.conv_6_2 = BasicConv3d(7, 8, kernel_size=4, padding='valid')
        # 8 x 14^3 -> 9 x 11^3
        self.conv_6_3 = BasicConv3d(8, 9, kernel_size=4, padding='valid')
        # 9 x 11^3 -> 10 x 8^3
        self.conv_6_4 = BasicConv3d(9, 10, kernel_size=4, padding='valid')
        # 10 x 8^3 -> 11 x 5^3
        self.conv_6_5 = BasicConv3d(10, 11, kernel_size=4, padding='valid')
        # 11 x 5^3 -> 12 x 2^3
        self.conv_6_6 = BasicConv3d(11, 12, kernel_size=4, padding='valid')

        # (9+6+4+2)=21 x 20^3 -> 5 x 16^3
        self.conv_7_1 = BasicConv3d(21, 5, kernel_size=5, padding='valid')
        # 5 x 16^3 -> 6 x 12^3
        self.conv_7_2 = BasicConv3d(5, 6, kernel_size=5, padding='valid')
        # 6 x 12^3 -> 7 x 8^3
        self.conv_7_3 = BasicConv3d(6, 7, kernel_size=5, padding='valid')
        # 7 x 8^3 -> 8 x 4^3
        self.conv_7_4 = BasicConv3d(7, 8, kernel_size=5, padding='valid')

        # (9+6+4+2)=21 x 20^3 -> 3 x 14^3
        self.conv_8_1 = BasicConv3d(21, 3, kernel_size=6, padding='valid')
        # 3 x 14^3 -> 4 x 8^3
        self.conv_8_2 = BasicConv3d(3, 4, kernel_size=6, padding='valid')
        # 4 x 8^3-> 5 x 2^3
        self.conv_8_3 = BasicConv3d(4, 5, kernel_size=6, padding='valid')

        # self.elu = nn.ELU(alpha=1.0)
        self.bn1 = nn.BatchNorm1d(290, eps=0.001, momentum=0.99)
        self.linear1 = nn.Linear(290, 256)
        self.bn2 = nn.BatchNorm1d(256, eps=0.001, momentum=0.99)
        self.linear2 = nn.Linear(546, 128)
        self.bn3 = nn.BatchNorm1d(128, eps=0.001, momentum=0.99)
        self.linear3 = nn.Linear(674, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        # 1 x 32^3 -> 4 x 30^3
        c_1 = self.conv_1_1(x)
        # 4 x 30^3 -> 4
        m_1_1 = torch.flatten(self.avgpool(c_1), 1)
        # 4 x 30^3 -> 5 x 28^3
        c_1 = self.conv_1_2(c_1)
        # 5 x 28^3 -> 5
        m_1_2 = torch.flatten(self.avgpool(c_1), 1)
        # 5 x 28^3 -> 6 x 26^3
        c_1 = self.conv_1_3(c_1)
        # 6 x 26^3 -> 6
        m_1_3 = torch.flatten(self.avgpool(c_1), 1)
        # 6 x 26^3 -> 7 x 24^3
        c_1 = self.conv_1_4(c_1)
        # 7 x 24^3 -> 7
        m_1_4 = torch.flatten(self.avgpool(c_1), 1)
        # 7 x 24^3 -> 8 x 22^3
        c_1 = self.conv_1_5(c_1)
        # 8 x 22^3 -> 8
        m_1_5 = torch.flatten(self.avgpool(c_1), 1)
        # 8 x 22^3 -> 9 x 20^3
        c_1 = self.conv_1_6(c_1)
        # 9 x 20^3-> 9
        m_1_6 = torch.flatten(self.avgpool(c_1), 1)

        # 1 x 32^3 -> 3 x 29^3
        c_2 = self.conv_2_1(x)
        # 3 x 29^3 -> 3
        m_2_1 = torch.flatten(self.avgpool(c_2), 1)
        # 3 x 29^3 -> 4 x 26^3
        c_2 = self.conv_2_2(c_2)
        # 4 x 26^3 -> 4
        m_2_2 = torch.flatten(self.avgpool(c_2), 1)
        # 4 x 26^3 -> 5 x 23^3
        c_2 = self.conv_2_3(c_2)
        # 5 x 23^3 -> 5
        m_2_3 = torch.flatten(self.avgpool(c_2), 1)
        # 5 x 23^3 -> 6 x 20^3
        c_2 = self.conv_2_4(c_2)
        # 6 x 20^3 -> 6
        m_2_4 = torch.flatten(self.avgpool(c_2), 1)

        # 1 x 32^3 -> 2 x 28^3
        c_3 = self.conv_3_1(x)
        # 2 x 28^3 -> 2
        m_3_1 = torch.flatten(self.avgpool(c_3), 1)
        # 2 x 28^3 -> 3 x 24^3
        c_3 = self.conv_3_2(c_3)
        # 3 x 16^3 -> 3
        m_3_2 = torch.flatten(self.avgpool(c_3), 1)
        # 3 x 24^3 -> 4 x 20^3
        c_3 = self.conv_3_3(c_3)
        # 4 x 20^3 -> 4
        m_3_3 = torch.flatten(self.avgpool(c_3), 1)

        # 1 x 32^3 -> 1 x 26^3
        c_4 = self.conv_4_1(x)
        # 1 x 26^3 -> 1
        m_4_1 = torch.flatten(self.avgpool(c_4), 1)
        # 1 x 26^3 -> 2 x 20^3
        c_4 = self.conv_4_2(c_4)
        # 2 x 20^3 -> 2
        m_4_2 = torch.flatten(self.avgpool(c_4), 1)

        # (9+6+4+2)=21 x 20^3
        x = torch.cat((c_1, c_2, c_3, c_4), dim=1)

        # 21 x 20^3 -> 10 x 18^3
        c_5 = self.conv_5_1(x)
        # 10 x 18^3 -> 10
        m_5_1 = torch.flatten(self.avgpool(c_5), 1)
        # 10 x 18^3 -> 11 x 16^3
        c_5 = self.conv_5_2(c_5)
        # 11 x 16^3 -> 11
        m_5_2 = torch.flatten(self.avgpool(c_5), 1)
        # 11 x 16^3 -> 12 x 14^3
        c_5 = self.conv_5_3(c_5)
        # 12 x 14^3 -> 12
        m_5_3 = torch.flatten(self.avgpool(c_5), 1)
        # 12 x 14^3 -> 13 x 12^3
        c_5 = self.conv_5_4(c_5)
        # 13 x 12^3 -> 13
        m_5_4 = torch.flatten(self.avgpool(c_5), 1)
        # 13 x 12^3 -> 14 x 10^3
        c_5 = self.conv_5_5(c_5)
        # 14 x 10^3 -> 14
        m_5_5 = torch.flatten(self.avgpool(c_5), 1)
        # 14 x 10^3 -> 15 x 8^3
        c_5 = self.conv_5_6(c_5)
        # 15 x 8^3 -> 15
        m_5_6 = torch.flatten(self.avgpool(c_5), 1)
        # 15 x 8^3 -> 16 x 6^3
        c_5 = self.conv_5_7(c_5)
        # 16 x 6^3 -> 16
        m_5_7 = torch.flatten(self.avgpool(c_5), 1)
        # 16 x 6^3 -> 17 x 4^3
        c_5 = self.conv_5_8(c_5)
        # 17 x 4^3 -> 17
        m_5_8 = torch.flatten(self.avgpool(c_5), 1)
        # 17 x 4^3 -> 18 x 2^3
        c_5 = self.conv_5_9(c_5)
        # 18 x 2^3 -> 18
        m_5_9 = torch.flatten(self.avgpool(c_5), 1)

        # 21 x 20^3 -> 7 x 17^3
        c_6 = self.conv_6_1(x)
        # 7 x 17^3 -> 7
        m_6_1 = torch.flatten(self.avgpool(c_6), 1)
        # 7 x 17^3 -> 8 x 14^3
        c_6 = self.conv_6_2(c_6)
        # 8 x 14^3 -> 8
        m_6_2 = torch.flatten(self.avgpool(c_6), 1)
        # 8 x 14^3 -> 9 x 11^3
        c_6 = self.conv_6_3(c_6)
        # 9 x 11^3 -> 9
        m_6_3 = torch.flatten(self.avgpool(c_6), 1)
        # 9 x 11^3-> 10 x 8^3
        c_6 = self.conv_6_4(c_6)
        # 10 x 8^3 -> 10
        m_6_4 = torch.flatten(self.avgpool(c_6), 1)
        # 10 x 8^3-> 11 x 5^3
        c_6 = self.conv_6_5(c_6)
        # 11 x 5^3 -> 11
        m_6_5 = torch.flatten(self.avgpool(c_6), 1)
        # 11 x 5^3-> 12 x 2^3
        c_6 = self.conv_6_6(c_6)
        # 12 x 2^3 -> 12
        m_6_6 = torch.flatten(self.avgpool(c_6), 1)

        # 21 x 20^3 -> 5 x 16^3
        c_7 = self.conv_7_1(x)
        # 5 x 16^3 -> 5
        m_7_1 = torch.flatten(self.avgpool(c_7), 1)
        # 5 x 16^3 -> 6 x 12^3
        c_7 = self.conv_7_2(c_7)
        # 6 x 12^3 -> 6
        m_7_2 = torch.flatten(self.avgpool(c_7), 1)
        # 6 x 12^3 -> 7 x 8^3
        c_7 = self.conv_7_3(c_7)
        # 7 x 8^3 -> 7
        m_7_3 = torch.flatten(self.avgpool(c_7), 1)
        # 7 x 8^3 -> 8 x 4^3
        c_7 = self.conv_7_4(c_7)
        # 8 x 4^3 -> 8
        m_7_4 = torch.flatten(self.avgpool(c_7), 1)

        # 21 x 20^3 -> 3 x 14^3
        c_8 = self.conv_8_1(x)
        # 3 x 14^3 -> 3
        m_8_1 = torch.flatten(self.avgpool(c_8), 1)
        # 3 x 14^3 -> 4 x 8^3
        c_8 = self.conv_8_2(c_8)
        # 4 x 8^3 -> 4
        m_8_2 = torch.flatten(self.avgpool(c_8), 1)
        # 4 x 8^3 -> 5 x 2^3
        c_8 = self.conv_8_3(c_8)
        # 5 x 2^3 -> 5
        m_8_3 = torch.flatten(self.avgpool(c_8), 1)

        # ((4+5+6+7+8+9)=39 + (3+4+5+6)=18 + (2+3+4)=9 + (1+2)=3 +
        # (10+11+12+13+14+15+16+17+18)=126 + (7+8+9+10+11+12)=57 + (5+6+7+8)=26 + (3+4+5)=12)=290
        m = torch.cat(
            (m_1_1, m_1_2, m_1_3, m_1_4, m_1_5, m_1_6,
             m_2_1, m_2_2, m_2_3, m_2_4,
             m_3_1, m_3_2, m_3_3,
             m_4_1, m_4_2,
             m_5_1, m_5_2, m_5_3, m_5_4, m_5_5, m_5_6, m_5_7, m_5_8, m_5_9,
             m_6_1, m_6_2, m_6_3, m_6_4, m_6_5, m_6_6,
             m_7_1, m_7_2, m_7_3, m_7_4,
             m_8_1, m_8_2, m_8_3),
            dim=1)

        fc1 = self.bn1(m)
        # 290 -> 256
        m = self.linear1(fc1)
        m = F.elu(m)
        fc2 = self.bn2(m)
        # 290+256=546
        fc2 = torch.cat((fc1, fc2), dim=1)
        # 546 -> 128
        m = self.linear2(fc2)
        m = F.elu(m)
        fc3 = self.bn3(m)
        # 546+128=674
        fc3 = torch.cat((fc2, fc3), dim=1)
        # 674 -> num_classes
        m = self.linear3(fc3)

        return m


if __name__ == "__main__":
    model = YOPO(num_classes=10)
    input = torch.randn(10, 1, 32, 32, 32)
    output = model(input)
    print(output.shape)
    print(model)
    print(list(model.modules()))
