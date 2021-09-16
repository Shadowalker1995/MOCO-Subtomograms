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
        D = 24
        H = 24
        W = 24
        self.keepfc = keepfc

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # 1 x 24^3 -> 4 x 22^3
        self.conv_1_1 = BasicConv3d(1, 4, kernel_size=3, padding='valid')
        # 4 x 22^3 -> 5 x 20^3
        self.conv_1_2 = BasicConv3d(4, 5, kernel_size=3, padding='valid')
        # 5 x 20^3 -> 6 x 18^3
        self.conv_1_3 = BasicConv3d(5, 6, kernel_size=3, padding='valid')
        # 6 x 18^3 -> 7 x 16^3
        self.conv_1_4 = BasicConv3d(6, 7, kernel_size=3, padding='valid')
        # 7 x 16^3 -> 8 x 14^3
        self.conv_1_5 = BasicConv3d(7, 8, kernel_size=3, padding='valid')
        # 8 x 14^3 -> 9 x 12^3
        self.conv_1_6 = BasicConv3d(8, 9, kernel_size=3, padding='valid')

        # 1 x 24^3 -> 3 x 21^3
        self.conv_2_1 = BasicConv3d(1, 3, kernel_size=4, padding='valid')
        # 3 x 21^3 -> 4 x 18^3
        self.conv_2_2 = BasicConv3d(3, 4, kernel_size=4, padding='valid')
        # 4 x 18^3 -> 5 x 15^3
        self.conv_2_3 = BasicConv3d(4, 5, kernel_size=4, padding='valid')
        # 5 x 15^3 -> 6 x 12^3
        self.conv_2_4 = BasicConv3d(5, 6, kernel_size=4, padding='valid')

        # 1 x 24^3 -> 2 x 20^3
        self.conv_3_1 = BasicConv3d(1, 2, kernel_size=5, padding='valid')
        # 2 x 20^3 -> 3 x 16^3
        self.conv_3_2 = BasicConv3d(2, 3, kernel_size=5, padding='valid')
        # 3 x 20^3 -> 4 x 12^3
        self.conv_3_3 = BasicConv3d(3, 4, kernel_size=5, padding='valid')

        # 1 x 24^3 -> 1 x 18^3
        self.conv_4_1 = BasicConv3d(1, 1, kernel_size=7, padding='valid')
        # 1 x 18^3 -> 2 x 12^3
        self.conv_4_2 = BasicConv3d(1, 2, kernel_size=7, padding='valid')

        # (9+6+4+2)=21 x 12^3 -> 10 x 10^3
        self.conv_5_1 = BasicConv3d(21, 10, kernel_size=3, padding='valid')
        # 10 x 10^3 -> 11 x 8^3
        self.conv_5_2 = BasicConv3d(10, 11, kernel_size=3, padding='valid')
        # 11 x 8^3 -> 12 x 6^3
        self.conv_5_3 = BasicConv3d(11, 12, kernel_size=3, padding='valid')
        # 12 x 6^3 -> 13 x 4^3
        self.conv_5_4 = BasicConv3d(12, 13, kernel_size=3, padding='valid')
        # 13 x 4^3 -> 14 x 2^3
        self.conv_5_5 = BasicConv3d(13, 14, kernel_size=3, padding='valid')

        # (9+6+4+2)=21 x 12^3 -> 7 x 9^3
        self.conv_6_1 = BasicConv3d(21, 7, kernel_size=4, padding='valid')
        # 7 x 9^3 -> 8 x 6^3
        self.conv_6_2 = BasicConv3d(7, 8, kernel_size=4, padding='valid')
        # 8 x 6^3 -> 9 x 3^3
        self.conv_6_3 = BasicConv3d(8, 9, kernel_size=4, padding='valid')

        # (9+6+4+2)=21 x 12^3 -> 5 x 8^3
        self.conv_7_1 = BasicConv3d(21, 5, kernel_size=5, padding='valid')
        # 5 x 8^3 -> 6 x 4^3
        self.conv_7_2 = BasicConv3d(5, 6, kernel_size=5, padding='valid')

        # (9+6+4+2)=21 x 12^3 -> 3 x 7^3
        self.conv_8_1 = BasicConv3d(21, 3, kernel_size=6, padding='valid')
        # 3 x 7^3 -> 4 x 2^3
        self.conv_8_2 = BasicConv3d(3, 4, kernel_size=6, padding='valid')

        # self.elu = nn.ELU(alpha=1.0)
        self.bn1 = nn.BatchNorm1d(171, eps=0.001, momentum=0.99)
        self.linear1 = nn.Linear(171, 128)
        self.bn2 = nn.BatchNorm1d(128, eps=0.001, momentum=0.99)
        self.linear2 = nn.Linear(299, 64)
        self.bn3 = nn.BatchNorm1d(64, eps=0.001, momentum=0.99)
        self.linear3 = nn.Linear(363, num_classes)

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
        # 1 x 24^3 -> 4 x 22^3
        c_1 = self.conv_1_1(x)
        # 4 x 22^3 -> 4
        m_1_1 = torch.flatten(self.avgpool(c_1), 1)
        # 4 x 22^3 -> 5 x 20^3
        c_1 = self.conv_1_2(c_1)
        # 5 x 20^3 -> 5
        m_1_2 = torch.flatten(self.avgpool(c_1), 1)
        # 5 x 20^3 -> 6 x 18^3
        c_1 = self.conv_1_3(c_1)
        # 6 x 18^3 -> 6
        m_1_3 = torch.flatten(self.avgpool(c_1), 1)
        # 6 x 18^3 -> 7 x 16^3
        c_1 = self.conv_1_4(c_1)
        # 7 x 16^3 -> 7
        m_1_4 = torch.flatten(self.avgpool(c_1), 1)
        # 7 x 16^3 -> 8 x 14^3
        c_1 = self.conv_1_5(c_1)
        # 8 x 14^3 -> 8
        m_1_5 = torch.flatten(self.avgpool(c_1), 1)
        # 8 x 14^3 -> 9 x 12^3
        c_1 = self.conv_1_6(c_1)
        # 9 x 12^3-> 9
        m_1_6 = torch.flatten(self.avgpool(c_1), 1)

        # 1 x 24^3 -> 3 x 21^3
        c_2 = self.conv_2_1(x)
        # 3 x 21^3 -> 3
        m_2_1 = torch.flatten(self.avgpool(c_2), 1)
        # 3 x 21^3 -> 4 x 18^3
        c_2 = self.conv_2_2(c_2)
        # 4 x 18^3 -> 4
        m_2_2 = torch.flatten(self.avgpool(c_2), 1)
        # 4 x 18^3 -> 5 x 15^3
        c_2 = self.conv_2_3(c_2)
        # 5 x 15^3 -> 5
        m_2_3 = torch.flatten(self.avgpool(c_2), 1)
        # 5 x 15^3 -> 6 x 12^3
        c_2 = self.conv_2_4(c_2)
        # 6 x 12^3 -> 6
        m_2_4 = torch.flatten(self.avgpool(c_2), 1)

        # 1 x 24^3 -> 2 x 20^3
        c_3 = self.conv_3_1(x)
        # 2 x 20^3 -> 2
        m_3_1 = torch.flatten(self.avgpool(c_3), 1)
        # 2 x 20^3 -> 3 x 16^3
        c_3 = self.conv_3_2(c_3)
        # 3 x 16^3 -> 3
        m_3_2 = torch.flatten(self.avgpool(c_3), 1)
        # 3 x 16^3 -> 4 x 12^3
        c_3 = self.conv_3_3(c_3)
        # 4 x 12^3 -> 4
        m_3_3 = torch.flatten(self.avgpool(c_3), 1)

        # 1 x 24^3 -> 1 x 18^3
        c_4 = self.conv_4_1(x)
        # 1 x 18^3 -> 1
        m_4_1 = torch.flatten(self.avgpool(c_4), 1)
        # 1 x 18^3 -> 2 x 12^3
        c_4 = self.conv_4_2(c_4)
        # 2 x 12^3 -> 2
        m_4_2 = torch.flatten(self.avgpool(c_4), 1)

        # (9+6+4+2)=21 x 12^3
        x = torch.cat((c_1, c_2, c_3, c_4), dim=1)

        # 21 x 12^3 -> 10 x 10^3
        c_5 = self.conv_5_1(x)
        # 10 x 10^3 -> 10
        m_5_1 = torch.flatten(self.avgpool(c_5), 1)
        # 10 x 10^3 -> 11 x 8^3
        c_5 = self.conv_5_2(c_5)
        # 11 x 8^3 -> 11
        m_5_2 = torch.flatten(self.avgpool(c_5), 1)
        # 11 x 8^3 -> 12 x 6^3
        c_5 = self.conv_5_3(c_5)
        # 12 x 6^3 -> 12
        m_5_3 = torch.flatten(self.avgpool(c_5), 1)
        # 12 x 6^3 -> 13 x 4^3
        c_5 = self.conv_5_4(c_5)
        # 13 x 4^3 -> 13
        m_5_4 = torch.flatten(self.avgpool(c_5), 1)
        # 13 x 4^3 -> 14 x 2^3
        c_5 = self.conv_5_5(c_5)
        # 14 x 2^3 -> 14
        m_5_5 = torch.flatten(self.avgpool(c_5), 1)

        # 21 x 12^3 -> 7 x 9^3
        c_6 = self.conv_6_1(x)
        # 7 x 9^3 -> 7
        m_6_1 = torch.flatten(self.avgpool(c_6), 1)
        # 7 x 9^3 -> 8 x 6^3
        c_6 = self.conv_6_2(c_6)
        # 8 x 6^3 -> 8
        m_6_2 = torch.flatten(self.avgpool(c_6), 1)
        # 8 x 6^3 -> 9 x 3^3
        c_6 = self.conv_6_3(c_6)
        # 9 x 3^3 -> 9
        m_6_3 = torch.flatten(self.avgpool(c_6), 1)

        # 21 x 12^3 -> 5 x 8^3
        c_7 = self.conv_7_1(x)
        # 5 x 8^3 -> 5
        m_7_1 = torch.flatten(self.avgpool(c_7), 1)
        # 5 x 8^3 -> 6 x 4^3
        c_7 = self.conv_7_2(c_7)
        # 6 x 4^3 -> 6
        m_7_2 = torch.flatten(self.avgpool(c_7), 1)

        # 21 x 12^3 -> 3 x 7^3
        c_8 = self.conv_8_1(x)
        # 3 x 7^3 -> 3
        m_8_1 = torch.flatten(self.avgpool(c_8), 1)
        # 3 x 7^3 -> 4 x 2^3
        c_8 = self.conv_8_2(c_8)
        # 4 x 2^3 -> 4
        m_8_2 = torch.flatten(self.avgpool(c_8), 1)

        # ((4+5+6+7+8+9)=39 + (3+4+5+6)=18 + (2+3+4)=9 + (1+2)=3 +
        # (10+11+12+13+14)=60 + (7+8+9)=24 + (5+6)=11 + (3+4)=7)=171 x 1^3
        m = torch.cat(
            (m_1_1, m_1_2, m_1_3, m_1_4, m_1_5, m_1_6,
             m_2_1, m_2_2, m_2_3, m_2_4,
             m_3_1, m_3_2, m_3_3,
             m_4_1, m_4_2,
             m_5_1, m_5_2, m_5_3, m_5_4, m_5_5,
             m_6_1, m_6_2, m_6_3,
             m_7_1, m_7_2,
             m_8_1, m_8_2),
            dim=1)

        fc1 = self.bn1(m)
        # 171 -> 128
        m = self.linear1(fc1)
        m = F.elu(m)
        fc2 = self.bn2(m)
        # 171+128=299
        fc2 = torch.cat((fc1, fc2), dim=1)
        # 299 -> 64
        m = self.linear2(fc2)
        m = F.elu(m)
        fc3 = self.bn3(m)
        # 299+64=363
        fc3 = torch.cat((fc2, fc3), dim=1)
        # 363 -> num_classes
        m = self.linear3(fc3)

        return m


if __name__ == "__main__":
    model = YOPO(num_classes=10)
    input = torch.randn(10, 1, 24, 24, 24)
    output = model(input)
    print(output.shape)
    print(model)
    print(list(model.modules()))
