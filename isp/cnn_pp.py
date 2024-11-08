#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
#对应common.py中extract_parameters_2

import torch
import torch.nn as nn
from yolox.models.network_blocks import *  #有DWConv和BaseConv
from isp.util_filters import lrelu

class CNN_PP(nn.Module):
    """
    CNN-PP module. 
    """
    def __init__(
        self,
        #channels,
        depthwise=False,
        #in_channels=[256, 512, 1024],
        act="lrelu", #激活函数
    ):
        super().__init__() #继承父类的__init__()
        output_dim = 15 #DIP模块参数量15个
        channels = 16
        Conv = DWConv if depthwise else BaseConv #可以选择使用不同卷积
        self.ex_conv0 = Conv(
            3, channels, 3, 2, act=act
        )
        self.ex_conv1 = Conv(
            channels, 2*channels, 3, 2, act=act
        )
        self.ex_conv2 = Conv(
           2*channels, 2*channels, 3, 2, act=act
        )
        self.ex_conv3 = Conv(
            2*channels, 2*channels, 3, 2, act=act
        )
        self.ex_conv4 = Conv(
            2*channels, 2*channels, 3, 2, act=act
        )
        self.linear0 = nn.Linear(2048, 64)
        #self.act_ex = get_activation(act, inplace=True) #看看源代码中的激活函数
        self.linear1 = nn.Linear(64, output_dim) 
        #self.relu1 = nn.LeakyReLU(inplace=True)
    
    def forward(self, input_256_256): #输入一个batch的图像
        """
        Args:
            inputs: input images 256*256.

        Returns:
            Tuple[Tensor]: fliter_features.
        """
        x = self.ex_conv0(input_256_256)
        x = self.ex_conv1(x)
        x = self.ex_conv2(x)
        x = self.ex_conv3(x)
        x = self.ex_conv4(x)
        x = torch.reshape(x, [-1, 2048]) #shape中-1表示这个维度的大小，程序运行时会自动计算填充（因为变换前后元素数量不变，我们可以根据其他维度的大小，最终确定-1这个位置应该表示的数字）

        x = self.linear0(x)
        x = lrelu(x)
        fliter_features = self.linear1(x) #DIP的15个参数
        return fliter_features
