# encoding: utf-8
import os

import torch
import torch.nn as nn #修改
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp.yolox_base_ssl import Exp as MyExp #yolox_base.py中的Exp


class Exp(MyExp): #继承yolox_base.py中的Exp
    def __init__(self):
        super(Exp, self).__init__()
        
        self.num_classes = 4 #此处修改
        
        #------------------nano--------------------#
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416) #将图像缩放为416*416
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (416, 416)
        #self.mosaic_prob = 0.5
        self.enable_mixup = False
        #self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        #------------------nano--------------------#
        
        self.warmup_epochs = 1
        
        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
    
    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOPAFPN
            from yolox.models.yolox_ssl import YOLOX
            from yolox.models.yolo_head_ssl import YOLOXHead
            
            from pelee.peleenet import PeleeNet #1
            
            in_channels = [256, 512, 1024]   

            #-------------------------CNN-PP----------------------#
            #cnn_pp = CNN_PP() #2 #该类的实例化不需要传参数
            #-----------------------------------------------------#

            #------------------------PeleeNet---------------------#
            peleenet = PeleeNet() #2 #该类的实例化不需要传参数
            #-----------------------------------------------------#

            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            #self.model = YOLOX(backbone, head)
            #self.model = YOLOX(cnn_pp, backbone, head) #3
            self.model = YOLOX(peleenet, backbone, head) #3
            
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, aug_flag, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            VOCDetection,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            #MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        from yolox.data.datasets.mosaicdetection_ssl import MosaicDetection
        local_rank = get_local_rank() #0

        with wait_for_the_master(local_rank):
            dataset = VOCDetection(
                data_dir= "/home/21131213348/isp_yolox/YOLOX_peleenet_1_ssl_copy/underwater_dataset/train", #修改 #训练集的绝对路径
                #data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
                image_sets=[('train')], #修改
                #image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                img_size=self.input_size, #(416,416)
                preproc=TrainTransform(
                    max_labels=100, #最多的目标数量（可以改）
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            aug_flag,
            dataset,###
            mosaic=not no_aug, #true
            img_size=self.input_size, #(416,416)
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob, #0.5
                hsv_prob=self.hsv_prob), #1.0
            degrees=self.degrees, #10.0
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup, #False
            mosaic_prob=self.mosaic_prob, #1.0
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler( #无限采样器
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler( #取出一个batch
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs) #加载数据集

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import VOCDetection, ValTransform

        valdataset = VOCDetection(
            data_dir = "/home/21131213348/isp_yolox/YOLOX_peleenet_1_ssl_copy/underwater_dataset/train",
            #data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            #image_sets=[('test')],
            image_sets=[('val')], #验证集，会读取val.txt文件 #测试时可以改为"test"
            #image_sets=[('2007', 'test')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator


    def get_unlabeled_data_loader(self, aug_flag, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            VOCDetection,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            #MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        from yolox.data.datasets.mosaicdetection_ssl import MosaicDetection
        local_rank = get_local_rank() #0

        with wait_for_the_master(local_rank):
            dataset = VOCDetection( #类实例化，只运行init函数
                data_dir= "/home/21131213348/isp_yolox/YOLOX_peleenet_1_ssl_copy/underwater_dataset/train", #修改 #训练集的绝对路径
                #data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
                image_sets=[('train_unlabeled')], #建立一个unlabeled.txt文件
                #image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                img_size=self.input_size, #(416,416)
                preproc=TrainTransform(
                    max_labels=100, #最多的目标数量（可以改）
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            aug_flag,
            dataset,### 
            mosaic=not no_aug, #true
            img_size=self.input_size, #(416,416)
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob, #0.5
                hsv_prob=self.hsv_prob), #1.0
            degrees=self.degrees, #10.0
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup, #False
            mosaic_prob=self.mosaic_prob, #1.0
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler( #无限采样器
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler( #取出一个batch
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader_unlabeled = DataLoader(self.dataset, **dataloader_kwargs) #加载数据集

        return train_loader_unlabeled
