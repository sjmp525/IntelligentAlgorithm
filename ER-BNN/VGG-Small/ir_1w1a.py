import torch.nn as nn
import torch.nn.functional as F
from MyBinaryfunctionOnly import MyBiFunW, MyBiFunA
import torch
import math


class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

        self.MyBiFunW = MyBiFunW()
        self.MyBiFunA = MyBiFunA()

    def forward(self, input):
        w = self.weight
        a = input

        # bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        # bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        # sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()

        # bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        # ba = binaryfunction.BinaryQuantize().apply(a, self.k, self.t)

        bw = self.MyBiFunW(w)
        ba = self.MyBiFunA(a)
        
        # bw = bw * sw

        # 二值卷积 输出
        output = F.conv2d(ba, bw, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output