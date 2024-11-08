#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import numpy as np
from torchvision import transforms
import torch.nn as nn
import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES, voc_classes
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize_ssl import vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "--demo", default="image", type=str, help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument( #地址可以是放待处理图像的文件夹
        "--path", default="/home/zhouyue/yolox/YOLOX/000896.jpg",  type=str, help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="/home/zhouyue/yolox/YOLOX/exps/example/yolox_voc/yolox_voc_nano.py",
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default="/home/zhouyue/yolox/YOLOX/weights/best_ckpt_167736.pth", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf") ###置信度
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold") ###非极大抑制
    parser.add_argument("--tsize", default=640, type=int, help="test img size") ###测试图像的大小
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument( #与旧版本兼容
        "--legacy",
        dest="legacy",
        default=False,
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=voc_classes.VOC_CLASSES,
        #cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0} #存放图像信息的字典
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img) #字典加了一项后：{'id': 0, 'file_name': 'c000017.jpg'}
            img = cv2.imread(img) #读入图像
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2] #1080,1920
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img #{'id': 0, 'file_name': 'c000017.jpg', 'height': 1080, 'width': 1920, 'raw_img':...}

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio #0.333333
        
        #1.图像缩放为(640,640);2.图像格式转换cv2.imread读入(640,640,3)-->网络处理(3,640,640);3.图像数据类型转换
        img, _ = self.preproc(img, None, self.test_size) #(3, 640, 640)
        #torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
        img = torch.from_numpy(img).unsqueeze(0) #torch.Size([1, 3, 640, 640])
        img = img.float()
        if self.device == "gpu": #执行
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            #inps_256_256 = nn.functional.interpolate(img, size=(256,256)) #1 #将输入图像缩放为256*256 #不需要改了！！！           
            
            inps_304_304 = nn.functional.interpolate(img, size=(304,304))
            
            #outputs,image_afterprocess = self.model(inps_256_256,img) #2
            outputs,image_afterprocess = self.model(inps_304_304,img) #2 
            
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess( #后处理
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info, image_afterprocess

    def visual(self, image_name, output, img_info, cls_conf=0.35): #output:torch.Size([18, 7])
        ratio = img_info["ratio"] #0.3333333333333333
        img = img_info["raw_img"] #原图，没有resize
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4] #torch.Size([18, 4])

        # preprocessing: resize #将预测框大小还原到与原图对应！！！
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(image_name, img, bboxes, scores, cls, cls_conf, self.cls_names) #可视化结果：(1080, 1920, 3)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path): #判断图像地址是否是文件夹，是文件夹说明有多个图像
        files = get_image_list(path)
    else:
        files = [path] #['/home/zhouyue/yolox/YOLOX/assets/c000017.jpg']
    files.sort()
    for image_name in files: #遍历文件夹中的所有图像
        outputs, img_info, image_afterprocess= predictor.inference(image_name) #3
        result_image = predictor.visual(image_name, outputs[0], img_info, predictor.confthre)
        #result_image = predictor.visual(outputs[0], img_info, predictor.confthre) #可视化结果：(1080, 1920, 3)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        # ch = cv2.waitKey(0) # 设置 waitKey(0) , 则表示程序会无限制的等待用户的按键事件
        # if ch == 27 or ch == ord("q") or ch == ord("Q"): #按这几个键则停止for循环，按其他键则继续for循环，即继续预测文件夹中剩下的图像
        #     break

        #保存预处理后的图像###4
        image_array = image_afterprocess.cpu().numpy()       
        image_array = np.uint8(image_array)
        image_array = np.squeeze(image_array, 0)
        image_array = image_array.transpose(1,2,0)
        image_array = cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
        cv2.imwrite('/home/zhouyue/yolox/YOLOX/image1.jpg', image_array)
        #保存预处理后的图像###4



def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name) #/home/zhouyue/yolox/YOLOX/YOLOX_outputs/yolox_voc_nano
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result: #args.save_result=True
        vis_folder = os.path.join(file_name, "vis_res") #/home/zhouyue/yolox/YOLOX/YOLOX_outputs/yolox_voc_nano/vis_res
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt: #false
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf ###
    if args.nms is not None:
        exp.nmsthre = args.nms ###
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize) ###(640,640)

    model = exp.get_model()
    #logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu": #args.device=="gpu"
        model.cuda()
        if args.fp16: #false
            model.half()  # to FP16
    model.eval()

    if not args.trt: #false
        if args.ckpt is None: #不执行
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else: #执行
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse: #false
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt: #false
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, voc_classes.VOC_CLASSES, trt_file, decoder, #decoder = None
        args.device, args.fp16, args.legacy,
    )
    # predictor = Predictor(
    #     model, exp, COCO_CLASSES, trt_file, decoder,
    #     args.device, args.fp16, args.legacy,
    # )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name) #初始化yolox_voc_nano.py文件中的Exp类

    main(exp, args)
