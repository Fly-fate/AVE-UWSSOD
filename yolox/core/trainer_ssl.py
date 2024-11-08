#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import cv2

import torch.nn as nn
import datetime
import os
import time
from loguru import logger
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.exp.yolox_base_ssl import Exp
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
    load_peleenet_ckpt
)

def interleave(x, size): #交错：图像交错排列——l,u1,l,u2,1,u1... #x：torch.Size([960, 3, 32, 32])
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

class Trainer:
    def __init__(self, exp: Exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()
        
        #---------------------------------合并labeled和unlabeled-----------------------------------#
        inps_l, targets_l = self.prefetcher.next() #返回预处理之后的图像(数据增强mosaic、resize图像和标签) #它可以加速你的pytorch数据加载器 #self.prefetcher在before_train中定义 #self.prefetcher=DataPrefetcher(self.train_loader)     
        inps_u, targets_u = self.prefetcher_u.next() #直接用于求loss
        inps = interleave(torch.cat((inps_l, inps_u)), size=5) #交错 #size=(2+8)/2=5
        #targets =interleave( torch.cat((targets_l, targets_u)), 2*args.mu+1)
        #---------------------------------合并labeled和unlabeled-----------------------------------#

        # #保存预处理后的图像###4
        # img_list = torch.split(inps, 1, dim = 0)
        # for i in range(len(img_list)):   
        #     image = torch.squeeze(img_list[i])
        #     image = image.cpu().numpy()       
        #     image = np.uint8(image)
        #     #image = np.squeeze(image, 0)
        #     image = image.transpose(1,2,0)
        #     #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     path = '/home/zhouyue/yolox/YOLOX/' + str(i) +'.jpg'
        #     cv2.imwrite(path, image)
        #     #保存预处理后的图像###4

        inps = inps.to(self.data_type)
        targets_l = targets_l.to(self.data_type)
        targets_u = targets_u.to(self.data_type)
        targets_l.requires_grad = False #关闭求导
        targets_u.requires_grad = False #关闭求导
        
        inps_304_304 = nn.functional.interpolate(inps, size=(304,304)) #1 #第一条路
        #inps_256_256 = nn.functional.interpolate(inps, size=(256,256)) #1 #将输入图像缩放为256*256
        inps, targets_l, targets_u = self.exp.preprocess(inps, targets_l, targets_u, self.input_size) #改变输入图像大小，与多尺度训练有关 #第二条路
        #print(inps_256_256.shape,inps.shape) #打印出形状 #2
        data_end_time = time.time()
        

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            #outputs = self.model(inps, targets) #调用YOLOX类，inps, targets为YOLOX类forward的输入
            #outputs = self.model(inps_256_256, inps, targets) ######跑forward函数 #3
            #outputs = self.model(inps_304_304, inps, targets) ######跑forward函数 #2
            outputs = self.model(inps_304_304, inps, targets_l, targets_u) ######跑forward函数 #2

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward() #根据loss进行反向传播
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update( #计时
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        # logger.info(
        #     "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        # )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)
        
        #------------------------------------labeled-----------------------------------#
        # data related init
        #self.no_aug = True
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs #self.exp.no_aug_epochs=15

        #self.aug_flag_labeled = "default"
        self.train_loader = self.exp.get_data_loader( #加载好的数据集
            aug_flag='default', #default
            batch_size=self.args.batch_size, 
            is_distributed=self.is_distributed, #分布式训练
            no_aug=self.no_aug, #数据增强
            cache_img=self.args.cache, #缓存有关
        )
        #------------------------------------labeled-----------------------------------#
        
        #----------------------------------unlabeled-----------------------------------#
        #self.aug_flag_unlabeled = "strong"
        self.train_loader_unlabeled = self.exp.get_unlabeled_data_loader( #加载unlabeled data
            aug_flag='strong', #strong
            batch_size=self.args.batch_size_unlabeled, #unlabeled data的batchsize #在args里加入batch_size_unlabeled
            is_distributed=self.is_distributed, #分布式训练
            no_aug=self.no_aug, #数据增强
            cache_img=self.args.cache, #缓存有关
        )
        #----------------------------------unlabeled-----------------------------------#

        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader) #里面执行了很多图像预处理，执行了一次mosaic #整个训练过程中只执行一次
        self.prefetcher_u = DataPrefetcher(self.train_loader_unlabeled)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug: #如果没有mosaic
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            self.train_loader_unlabeled.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug: #如果没有关闭mosaic则保存权重
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({
                        "train/lr": self.meter["lr"].latest
                    })
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt #权重地址
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"] #self.device是gpu
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0
        #----------------------------加载peleenet的权重--------------------------#    
        #peleenet_ckpt_dir = '/home/21131213348/isp_yolox/YOLOX_peleenet_1/weights/peleenet.pth' #1
        peleenet_ckpt_dir = '/home/21131213348/isp_yolox/YOLOX_peleenet_1/weights/Pelee_VOC.pth'
        peleenet_ckpt = torch.load(peleenet_ckpt_dir, map_location=self.device) #pth文件中只有权重，没有其他信息
        model = load_peleenet_ckpt(model, peleenet_ckpt)
        #-----------------------------------------------------------------------#
        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "train/epoch": self.epoch + 1,
                })
                self.wandb_logger.log_images(predictions)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "curr_ap": ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "best_ap": self.best_ap,
                        "curr_ap": ap
                    }
                )
