#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp.yolox_base_ssl import Exp
from yolox.exp import get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices
#import wandb

#wandb.init(project="YOLOX", entity="zhouyue")

import sys
print(sys.path) #打印sys.path

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")   
    parser.add_argument(
        "-d", "--devices", default=0, type=int, help="device for training"
    )
    # parser.add_argument(
    #     "-d", "--devices", default=None, type=int, help="device for training"
    # )
    parser.add_argument(
        "-f",
        "--exp_file",
        default="/home/21131213348/isp_yolox/YOLOX_peleenet_1_ssl_copy/exps/example/yolox_voc/yolox_voc_nano_ssl.py",
        type=str,
        help="plz input your experiment description file",
    )
    # parser.add_argument(
    #     "-f",
    #     "--exp_file",
    #     default=None,
    #     type=str,
    #     help="plz input your experiment description file",
    # )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    #parser.add_argument("-c", "--ckpt", default="/home/zhouyue/yolox/YOLOX/weights/yolox_cspdarknet_nano.pth", type=str, help="checkpoint file")
    parser.add_argument("-c", "--ckpt", default="/home/21131213348/isp_yolox/YOLOX_peleenet_1_ssl_copy/weights/yolox_nano.pth", type=str, help="checkpoint file")
    #parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
        Implemented loggers include `tensorboard` and `wandb`.",
        #default="wandb"
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    #-------------------------------------ssl----------------------------------------------#
    #parser.add_argument("--mu", type=int, default=1, help="unlabeled是labeled的mu倍")
    #parser.add_argument("--WU", type=int, default=1, help="unlabeled loss的权重") #在yolo_head_ssl.py中设置了
    parser.add_argument("-u", "--batch-size-unlabeled", type=int, default=2, help="unlabeled batch size")
    #--------------------------------------------------------------------------------------#
    # parser.add_argument("--project", type=str, default="YOLOX", help="project name")
    # parser.add_argument("--entity", type=str, default="zhouyue", help="")

    return parser


@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
