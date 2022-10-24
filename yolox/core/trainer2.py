#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger

import torch
import cv2


from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)

import datetime
import os
import time


class Trainer2:
    def __init__(self, exp, args):
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
        self.local_rank = args.local_rank
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

    def train(self):
        self.before_train()

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)

        # data related init
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=False,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        # self.prefetcher = DataPrefetcher(self.train_loader)

        logger.info("Selecting image...")
        # img = cv2.imread("D:\\PyCharmProject\\ByteTrack-main\\datasets\\mix_crowdhuman_ch\\crowdhuman_train\\273271,1a0d6000b9e1f5b7.jpg")
        # cv2.imshow('image', img)
        # inps, targets = self.prefetcher.next()
        for iter_id, batch in enumerate(self.train_loader):
            img, label, _, _ = batch
            img = torch.permute(img, (0,2,3,1))
            img = img.numpy()[0]
            cv2.imshow('image', img)
            cv2.waitKey(0)
            break




