import torch
from torch import Tensor
import torch.nn as nn
import torch.fft as fft
import math
from modules.quaternion_ops import *
from modules.quaternion_layers import *


class Qsharpen(nn.Module):
    def __init__(self):
        super(Qsharpen, self).__init__()
        self.up = UP()
        self.mid = MID()
        self.down = DOWN()
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up3 = nn.Upsample(scale_factor=4)
        self.f1 = MFEM()
        self.f2 = MFEM()
        self.f3 = MFEM()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.Q_MS = QCM()

    def forward(self, Input1: Tensor, Input2: Tensor):
        # print(Input1.shape, Input2.shape)
        x = self.up1(Input1)
        x = self.Q_MS(x)
        z0 = self.up(Input1, Input2)
        z1 = self.mid(Input1, Input2)
        z2 = self.down(Input1, Input2)
        y0 = self.f1(z0, x)
        z1 = self.up2(z1)
        y1 = self.f2(z1, x)
        z2 = self.up3(z2)
        y2 = self.f3(z2, x)
        out_term = torch.cat((y0, y1, y2), dim=1)
        out_term = self.conv(out_term)
        return out_term


class UP(nn.Module):
    def __init__(self):
        super(UP, self).__init__()
        dim_l = 128
        self.up1 = nn.Upsample(scale_factor=4)
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.FM = cross_modal(4,4)

    def forward(self, Input1: Tensor, Input2: Tensor):
        x1 = self.up1(Input1)
        Input2 = self.conv(Input2)
        z0 = self.FM(x1, Input2)
        return z0


class MID(nn.Module):
    def __init__(self):
        super(MID, self).__init__()
        dim_l = 128
        self.down2 = nn.MaxPool2d(2)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.FM = cross_modal(4,4)

    def forward(self, Input1: Tensor, Input2: Tensor):
        x1 = self.up2(Input1)
        x2 = self.down2(Input2)
        x2 = self.conv(x2)
        z1 = self.FM(x1, x2)
        return z1


class DOWN(nn.Module):
    def __init__(self):
        super(DOWN, self).__init__()
        dim_l = 64
        self.down1 = nn.MaxPool2d(4)
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.FM = cross_modal(4,4)

    def forward(self, Input1: Tensor, Input2: Tensor):
        x2 = self.down1(Input2)
        x2 = self.conv(x2)
        z2 = self.FM(Input1, x2)
        return z2


class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        self.conv4 = nn.Sequential(
             nn.Conv2d(in_channels=36, out_channels=16, kernel_size=3, stride=1, padding=1),  # 32+pan的一个通道
             nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input1, input2):
        z = torch.cat((input1, input2), dim=1)
        z = self.conv4(z)
        z = torch.sigmoid(z)
        return z


class MFE(nn.Module):
    def __init__(self, in_channels = 32, out_channels = 32, kernel_size1=1, kernel_size2=3, kernel_size3=5, groups=4):
        super(MFE, self).__init__()
        self.groups = groups
        self.conv = nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
                               kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
                               kernel_size=kernel_size1, padding=kernel_size1 // 2)
        self.conv2 = nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
                               kernel_size=kernel_size2, padding=kernel_size2 // 2)
        self.conv3 = nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
                               kernel_size=kernel_size3, padding=kernel_size3 // 2)
        self.cb1 = CB1()
        self.cb2 = CB1()
        self.cb3 = CB1()
        self.cb4 = CB1()
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, self.groups, 1)
        y1 = self.conv(x1)
        y1 = self.cb1(y1)
        y2 = self.conv(x2)
        y2 = self.conv1(y2)
        y2 = self.cb2(y2)
        y3 = self.conv(x3)
        y3 = self.conv2(y3)
        y3 = self.cb3(y3)
        y4 = self.conv(x4)
        y4 = self.conv3(y4)
        y4 = self.cb4(y4)
        z = torch.cat((y1, y2, y3, y4), dim=1)
        y4 = self.conv6(z)
        return y4


class basicBlock_dual(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(basicBlock_dual, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out)
        return out


class cross_modal(nn.Module):   # existing error
    def __init__(self, in_channels, out_channels):
        super(cross_modal, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basic1 = basicBlock_dual(in_channels=self.in_channels, out_channels=1)
        self.basic2 = basicBlock_dual(in_channels=self.in_channels, out_channels=1)
        self.basic3 = basicBlock_dual(in_channels=self.in_channels, out_channels=1)
        self.conv1x1=nn.Conv2d(in_channels=2*self.in_channels,out_channels=self.out_channels,kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x11 = self.basic1(x1)
        x1_attn1 = self.sigmoid(x11)
        x1_intra = torch.mul(x1_attn1, x1)
        x21 = self.basic2(x2)
        x2_attn1 = self.sigmoid(x21)
        x2_intra = torch.mul(x2_attn1, x2)

        x3=x1+x2
        x31=self.basic3(x3)
        x3_attn1 = self.sigmoid(x31)

        x12 = torch.mul(x3_attn1, x1_intra)
        x1_out=x1+x12
        x22 = torch.mul(x3_attn1, x2_intra)
        x2_out = x2 + x22

        x_c = torch.cat((x1_out, x2_out), dim=1)
        out=self.conv1x1(x_c)
        return out


class MFEM(nn.Module):
    def __init__(self):
        super(MFEM, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.cam = CAM(32)
        self.up2 = nn.Upsample(scale_factor=2)
        self.mfe = MFE()
        self.sab = SAB()

    def forward(self, Input2, ms):
        y0 = self.conv0(Input2)
        y1 = self.cam(y0)
        # y1 = self.up2(y1)
        # print(y1.shape)
        y2 = self.mfe(y1)
        # y2 = self.up2(y2)
        # print(y0.shape, ms.shape, y2.shape, y1.shape)
        y2 = self.sab(y2, ms)
        y3 = self.sab(y1, ms)
        z = torch.cat((y2, y3), dim=1)
        return z


class CAM(nn.Module):
    def __init__(self, channel, reduction_ratio=4):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction_ratio), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CB1(nn.Module):
    def __init__(self, groups=1, norm_layer=None):
        super(CB1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')

        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(16)
        self.relu = nn.LeakyReLU(0.2)  # nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.bn2 = norm_layer(8)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class QCM(nn.Module):
    def __init__(self):
        super(QCM, self).__init__()
        self.conv1 = nn.Sequential(
            QuaternionConv(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            QuaternionConv(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(256)
        self.conv2 = nn.Sequential(
            QuaternionConv(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            QuaternionConv(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        y = self.conv1(x)
        y1 = self.avg_pool(y)
        # print(y1.shape)
        y2 = self.conv2(y1)
        y3 = y * y2
        z = y3 + x
        # print(z.shape)
        return z


if  __name__ == "__main__":
    dc = Qsharpen()
    A = torch.FloatTensor(size=(1, 4, 64, 64)).normal_(0, 1)
    B = torch.FloatTensor(size=(1, 1, 256, 256)).normal_(0, 1)
    out = dc(A, B)
    # print()