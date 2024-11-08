#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn
import numpy as np
import torch
import cv2
import math

#from yolox.models.cnn_pp import CNN_PP

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from isp.dip_process_nodefog import DIP
#from isp.dip_process import DIP


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    #def __init__(self, backbone=None, head=None):
    #def __init__(self, cnn_pp=None, backbone=None, head=None): #1
    def __init__(self, peleenet=None, backbone=None, head=None): #1
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        #self.cnn_pp = cnn_pp #2
        self.peleenet = peleenet #2
        self.backbone = backbone
        self.head = head
    

    #def forward(self, input, targets=None):
    #def forward(self, input_256_256, input, targets=None): #1 #input是一个batch的输入图像
    def forward(self, input_304_304, input, targets_l=None, targets_u=None): #1 #input是一个batch的输入图像 #targets是标签

        #---------------------------peleenet---------------------------#
        filter_features = self.peleenet(input_304_304) #2
        #--------------------------------------------------------------#

        #----------------------------CNN_PP----------------------------#
        #filter_features = self.cnn_pp(input_256_256) #2 #得到filters的参数特征向量fliter_features #输入256*256的图像
        #--------------------------------------------------------------#
        
        #----------------------------------------------ulap-----------------------------------------------#
        #ULAP_process = ULAP(input) 
        #image_afterulap = ULAP_process.process(filter_features) #注意返回的图像的像素大小，要缩放为416*416
        #----------------------------------------------ulap-----------------------------------------------#

        #DIP 
        DIP_process = DIP(input) #4 #ulap时这里也要改

        # #将pytorch格式传入的图像(b,c,h,w)转换成tensorflow格式(b,h,w,c)
        # filter_input = np.transpose(input.cpu().numpy(),[0,2,3,1])        
        
        # dark = np.zeros((filter_input.shape[0], filter_input.shape[1], filter_input.shape[2]))

        # defog_A = np.zeros((filter_input.shape[0], filter_input.shape[3]))

        # IcA = np.zeros((filter_input.shape[0], filter_input.shape[1], filter_input.shape[2]))

        # for i in range(filter_input.shape[0]): #循环6次，遍历一个batch中的每一张图像
        #     #print(filter_input[i].shape)
        #     dark_i = DarkChannel(filter_input[i])
        #     defog_A_i = AtmLight(filter_input[i], dark_i)
        #     IcA_i = DarkIcA(filter_input[i], defog_A_i)
        #     dark[i, ...] = dark_i
        #     defog_A[i, ...] = defog_A_i
        #     IcA[i, ...] = IcA_i
        # IcA = np.expand_dims(IcA, axis=-1)

        
        # filter_input = torch.tensor(filter_input)
        # filter_input = filter_input.cuda() #转为gpu tensor

        # defog_A = torch.tensor(defog_A)
        # defog_A = defog_A.cuda()
        # defog_A = defog_A.to(torch.float32)

        # IcA = torch.tensor(IcA)
        # IcA = IcA.cuda()
        # IcA = IcA.to(torch.float32)

        #image_afterprocess = DIP_process.image_process(filter_input, filter_features) #5
        image_afterprocess = DIP_process.image_process(filter_features) #5 #注意返回的图像的像素大小，要缩放为416*416
        image_afterprocess = image_afterprocess.permute(0,3,1,2) #6


        # fpn output content features of [dark3, dark4, dark5]
        #fpn_outs = self.backbone(input)
        fpn_outs = self.backbone(image_afterprocess) #3 #输入416*416的图像

        if self.training:
            assert targets_l is not None
            assert targets_u is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets_l, targets_u, input #7 #input在求loss时用，使用未经DIP处理的图像
            )
            outputs = { #各种loss计算出来之后放在一个字典里
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs#, image_afterprocess

#一些函数
def DarkChannel(im):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
        return dc

def AtmLight(im, dark):
        [h, w] = im.shape[:2]
        imsz = h * w
        numpx = int(max(math.floor(imsz / 1000), 1))
        darkvec = dark.reshape(imsz, 1)
        imvec = im.reshape(imsz, 3)

        indices = darkvec.argsort(0)
        indices = indices[(imsz - numpx):imsz]

        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx
        return A  
def DarkIcA(im, A):
        im3 = np.empty(im.shape, im.dtype)
        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]
        return DarkChannel(im3)