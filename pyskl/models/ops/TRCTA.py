from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.ops import SqueezeExcitation
class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, input):
        return input.permute(self.dims).contiguous()
class MLP(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.BN1 = nn.BatchNorm1d( num_h, momentum=0.5)
        self.relu = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(num_h, num_h)
        self.relu2 = torch.nn.ReLU()
        self.BN2 = nn.BatchNorm1d(num_h, affine=True)

        self.linear3 = torch.nn.Linear(num_h, num_o)
        self.sigmoid = nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.BN3 = nn.BatchNorm1d(num_o, momentum=0.5)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.BN1(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.BN2(x)

        x = self.linear3(x)
        x = self.sigmoid(x)
        x = self.BN3(x)
        return x
class TTR(nn.Module):
    def __init__(self,inchannel=512, outchannel=512):
        super(TTR, self).__init__()
        self.p=Permute((0, 1, 3,2))  # 换维度
        self.pool = nn.AdaptiveAvgPool2d(( 1, 1))
        self.mlp = MLP(inchannel,256,inchannel)


        self.conv1x1 = nn.Conv3d(in_channels=inchannel, out_channels=outchannel, kernel_size=(1, 1, 1), stride=(1, 1, 1),padding=(0, 0, 0))






    def forward(self, input):
         #input 4 320 24 7 7(时间维度)     另一个模块结果4 128 7 7（空间维度）

        B, C, T, H, W = input.size()  # 4 320 24 7 7
        input1 = input.view(B,C,T,H*W)# 4 320  24 49
        p1 = self.p(input1)  # 4 320 49 24

        m1=torch.matmul(input1,p1)#4 320 24 24

        m2=self.pool(m1)# 4 512 1 1
        m2 = torch.flatten(m2, start_dim=1)  # 展并数据 4 512
        l1 = self.mlp(m2)  # 4 512

        l1 = l1.reshape(-1, 512, 1, 1)  # B 512 1 1
        l2=l1.unsqueeze(-1)#4 320 1 1 1
        l3=torch.mul(l2, input)#4 320 24 7 7
        l4=self.conv1x1(input)#4 320  24 7 7
        l5=l4+l3
        return l5
class CrossChannelFeatureAugment(nn.Module):
    """Cross-channel feature augmentation.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=10,
                 squeeze_factor=1):
        super().__init__()
        inter_channels = (out_channels // groups) * groups
        self.map1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.attend = SELayer(inter_channels, squeeze_factor, bias=False)
        self.group = DualGroupConv(
            inter_channels,
            inter_channels,
            kernel_size,
            stride,
            padding,
            groups,
            bias=False)
        self.map2 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)

    def forward(self, x):
        out = F.relu(self.map1(x), inplace=True)
        out = self.attend(out)
        out = self.group(out)
        out = F.relu(self.map2(out), inplace=True)
        return out
class DualGroupConv(nn.Module):
    """Dual grouped convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 groups, bias):
        super().__init__()
        assert out_channels % groups == 0
        assert groups % 2 == 0
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(1,1),
            groups=groups,
            bias=bias)
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            groups=groups // 2,
            bias=bias)
        self.conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            (5,5),
            stride=stride,
            padding=(2,2),
            groups=groups,
            bias=bias)

    def forward(self, input):
        out1 = F.relu(self.conv1(input), inplace=True)
        out2 = F.relu(self.conv2(input), inplace=True)
        out3 = F.relu(self.conv3(input), inplace=True)
        out = out1 + out2+out3
        return out
class SELayer(SqueezeExcitation):

    def __init__(self, input_channels, squeeze_factor=1, bias=True):
        squeeze_channels = input_channels // squeeze_factor
        super().__init__(input_channels, squeeze_channels)

        if not bias:
            self.fc1.register_parameter('bias', None)
            self.fc2.register_parameter('bias', None)
class CTA(nn.Module):
    def __init__(self, in_channels=512,groups=(10, 6),squeeze_factor=1):
        super(CTA, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        # self.in_channels=in_channels
        # self.out_channels = out_channels
        self.rel_channels = in_channels // 2#256
        self.conv1 = nn.Conv3d(in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels, self.rel_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.rel_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.cfa5 = CrossChannelFeatureAugment(
            self.rel_channels,
          in_channels, (3, 3),
            stride=1,
            padding=(1, 0),
            groups=groups[0],
            squeeze_factor=squeeze_factor)
        self.tanh = nn.Tanh()



    def forward(self, input):#4 512 24 6 6
        x1, x2 = self.conv1(input).mean(-3), self.conv2(input).mean(-3)#4 512 7 7
        y1 = self.tanh(x1.unsqueeze(-1).unsqueeze(-1) - x2.unsqueeze(-3).unsqueeze(-3))#4 512 6 6 6 6
        y1=y1.mean(2)#4 256 7 7 7
        y1=y1.mean(2)#4 256 6 6
        z1 = self.conv4(y1)*self.alpha +x1#4 256 7 7
        o1=self.cfa5(z1)#4 512 7 7
        o1=o1.unsqueeze(-3)*input
        return o1





