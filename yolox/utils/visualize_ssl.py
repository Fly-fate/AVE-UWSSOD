#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

__all__ = ["vis"]

# 增加换行符
def __indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def vis(image_name, img, boxes, scores, cls_ids, conf=0.5, class_names=None): #置信度阈值：conf=0.3
    #-------------------写入xml文件--------------------#
    root_annotation = ET.Element('annotation') #创建根节点'annotation'
    tree = ET.ElementTree(root_annotation) #创建文档
    element_frame = ET.Element('frame')
    element_frame.text = os.path.basename(image_name)[:-4] #图像名
    root_annotation.append(element_frame)
    #-------------------------------------------------#
    for i in range(len(boxes)): #遍历一张图像中预测出的所有目标框
        box = boxes[i] #一个目标框的四个坐标
        cls_id = int(cls_ids[i]) #类别
        score = scores[i] #该类别得分
        if score < conf: ###在此处使用了置信度阈值！！！
            continue #跳过本次循环体中剩下尚未执行的语句，立即进行下一次的循环条件判定，可以理解为只是中止(跳过)本次循环，接着开始下一次循环。
        x0 = int(box[0]) #xmin
        y0 = int(box[1]) #ymin
        x1 = int(box[2]) #xmax
        y1 = int(box[3]) #ymax
        
        #---------------------可视化-----------------------#
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2) #在img（原图）上画预测框

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        #------------------写入xml文件---------------------#
        element_object = ET.Element('object')
        root_annotation.append(element_object)
        
        element_name = ET.Element('name')
        element_name.text = class_names[cls_id] #类别名
        element_object.append(element_name)

        element_bndbox = ET.Element('bndbox')
        element_object.append(element_bndbox)

        element_xmin = ET.Element('xmin')
        element_xmin.text = str(x0)
        element_ymin = ET.Element('ymin')
        element_ymin.text = str(y0)
        element_xmax = ET.Element('xmax')
        element_xmax.text = str(x1)
        element_ymax = ET.Element('ymax')
        element_ymax.text = str(y1)
        element_bndbox.append(element_xmin)
        element_bndbox.append(element_ymin)
        element_bndbox.append(element_xmax)
        element_bndbox.append(element_ymax)
    
    element_size = ET.Element('size')
    root_annotation.append(element_size)

    element_width = ET.Element('width')
    element_width.text = str(img.shape[1])
    element_size.append(element_width)

    element_height = ET.Element('height')
    element_height.text = str(img.shape[0])
    element_size.append(element_height)


    __indent(root_annotation) # 增加换行符
    # tree.write('/home/zhouyue/yolox/YOLOX/Pseudo_label/' + os.path.basename(image_name)[:-4] + '.xml', 
    #             encoding='utf-8', xml_declaration=True)
    tree.write('/home/zhouyue/yolox/YOLOX/' + os.path.basename(image_name)[:-4] + '.xml', 
                encoding='utf-8', xml_declaration=True)

    return img #原图，shape:(1080, 1920, 3)


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
