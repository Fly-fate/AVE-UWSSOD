#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import torch

# def load_ckpt(model, ckpt):
#     model_state_dict = model.state_dict()
#     load_dict = {}
#     for key_model, v in model_state_dict.items():
#         if key_model not in ckpt:
#             logger.warning(
#                 "{} is not in the ckpt. Please double check and see if this is desired.".format(
#                     key_model
#                 )
#             )
#             continue
#         v_ckpt = ckpt[key_model]
#         if v.shape != v_ckpt.shape:
#             logger.warning(
#                 "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
#                     key_model, v_ckpt.shape, key_model, v.shape
#                 )
#             )
#             continue
#         load_dict[key_model] = v_ckpt

#     model.load_state_dict(load_dict, strict=False)
#     return model


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict() #model.state_dict()能够获取模型中的所有参数
    load_dict = {} #新的字典用来放需要加载的权重
    #---------------------------------检查与训练权重与模型是否匹配---------------------------------#
    for key_model, v in model_state_dict.items(): #遍历模型中所有参数，检查是否符合要求
        if key_model not in ckpt: #如果模型中的参数名和预训练权重不匹配
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        
        v_ckpt = ckpt[key_model] #预训练权重值

        #--------------------------------去掉因为类别不同而不匹配的参数------------------#
        # if "head.cls_preds" not in key_model: #如果该键与类别有关则不加载
        #     load_dict[key_model] = v_ckpt
        #     if v.shape != v_ckpt.shape:
        #         logger.warning( #模型中参数值类型和与训练权重不匹配
        #             "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
        #                 key_model, v_ckpt.shape, key_model, v.shape
        #             )
        #         )
        #         continue
        #load_dict[key_model] = v_ckpt #每个参数依次赋值
        #--------------------------------去掉因为类别不同而不匹配的参数------------------#

        #-----------------------------------backbone权重-------------------------------#
        if "backbone" in key_model: #只加载backbone权重
            load_dict[key_model] = v_ckpt
        #-----------------------------------backbone权重-------------------------------#

    #---------------------------------检查与训练权重与模型是否匹配---------------------------------#
    #print(load_dict.keys())
    model.load_state_dict(load_dict, strict=False) #给模型加载预训练权重
    print(load_dict.keys()) #打印出已经加载的权重
    print('Load backbone cpkt success')
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)



def load_peleenet_ckpt(model, ckpt_peleenet):
    model_state_dict = model.state_dict() #model.state_dict()能够获取模型中的所有参数
    load_dict = {} #新的字典用来放需要加载的权重
    #---------------------------------检查与训练权重与模型是否匹配---------------------------------#
    for key_model, v in model_state_dict.items(): #遍历模型中所有参数，检查是否符合要求
        origin_key_model = key_model #模型中key的名字
        key_model = key_model[9:] #权重文件中key的名字
        #key_model = key_model[18:] #2
        print(key_model)
        if key_model not in ckpt_peleenet: #如果模型中的参数名和预训练权重不匹配
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        
        v_ckpt = ckpt_peleenet[key_model] #预训练权重值

        #-----------------------------------backbone权重-------------------------------#
        if "features" in key_model: #只加载peleenet的backbone权重，其中带有features标志
            load_dict[origin_key_model] = v_ckpt #load_dict中使用模型中key的名字，因为要加载到模型中，不对应则无法加载
        #-----------------------------------backbone权重-------------------------------#
            
        #-----------------------------------backbone权重(peleenet.pth)-------------------------------#
        #load_dict[origin_key_model] = v_ckpt #load_dict中使用模型中key的名字，因为要加载到模型中，不对应则无法加载 #1
        #--------------------------------------------------------------------------------------------#

    #---------------------------------检查与训练权重与模型是否匹配---------------------------------#
    print(load_dict.keys())
    #model.load_state_dict(load_dict)
    model.load_state_dict(load_dict, strict=False) #给模型加载预训练权重
    #print(load_dict.keys()) #打印出已经加载的权重
    print('Load peleenet_backbone cpkt success')
    return model