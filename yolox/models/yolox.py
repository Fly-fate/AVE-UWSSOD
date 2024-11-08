#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn
import numpy as np
import torch
import cv2
import math

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
#from isp.dip_process import DIP
from isp.dip_process_nodefog import DIP

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    #def __init__(self, backbone=None, head=None):
    def __init__(self, peleenet=None, backbone=None, head=None): #1
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        self.peleenet = peleenet #2
        self.backbone = backbone
        self.head = head

    #def forward(self, x, targets=None):
    def forward(self, input_304_304, input, targets=None): #1 #input是一个batch的输入图像
        #---------------------------peleenet---------------------------#
        filter_features = self.peleenet(input_304_304) #2
        #--------------------------------------------------------------#
        
        #DIP 
        DIP_process = DIP(input) #4

        image_afterprocess = DIP_process.image_process(filter_features) #5 #注意返回的图像的像素大小，要缩放为416*416
        image_afterprocess = image_afterprocess.permute(0,3,1,2) #6

        # fpn output content features of [dark3, dark4, dark5]
        #fpn_outs = self.backbone(x)
        fpn_outs = self.backbone(image_afterprocess) #3 #输入416*416的图像

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, input #7
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
