#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import torch
import warnings

import numpy as np
import random

from time import sleep
from random import randint
from sklearn.model_selection import train_test_split
import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager
from torch.utils.data import DataLoader
import src.engine.fsl_dataset as DL
import pandas as pd
from launch import default_argument_parser, logging_train_setup
import PIL.Image as Image
from src.data.transforms import get_transforms
from collections import Counter
from pytorch_metric_learning import samplers
import cv2
warnings.filterwarnings("ignore")


class OcularDataset():
    def __init__(self, images, labels, args, preprocess, split):
        # images
        self.X = images
        # labels
        self.y = labels
        self.split = split
        self.path = '/mnt/data1/llr_data/AMD_Classification/ODIR/ODIR-5K_Training_Dataset'
        self.transform = preprocess

    def get_class_num(self):
        return 2

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self.split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self.split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num
        
        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return  [1.0] * cls_num #weight_list.tolist()

    def __len__(self):
        # return length of image samples
        return len(self.X)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.X.iloc[idx])
        image = Image.open(image_path)
        label = self.y[idx]    
        _img_name = image_path.split('/')[-1]
        # if self.split == 'train':
        #     image = image.resize((1024, 1024))
        #     try:
        #         mask = cv2.imread('./mask/ODIR/' +_img_name.split('.')[0]+'_mask0.jpg') 
        #         mask = cv2.resize(mask, (np.array(image).shape[1], np.array(image).shape[0]))
        #         print('mask', mask.shape)
        #     except:
        #         mask = np.zeros_like(image)
        
        #     # print('image shape:', np.array(image).shape)
        #     # print('name:', _img_name, 'mask shape:', mask.shape)
        #     image = Image.fromarray(np.concatenate([image, mask], axis=0))
            
        #     if self.transform is not None:
        #         image = self.transform(image)
        #     anco_sample = {'image': image, 'label': label, 'mask': mask, 'img_name': _img_name}
        # else:
        #     if self.transform is not None:
        #         image = self.transform(image)
        #     anco_sample = {'image': image, 'label': label}
        
        if self.transform is not None:
            image = self.transform(image)
            anco_sample = {'image': image, 'label': label}
        
        return anco_sample #(data, label)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist
    # cfg.DIST_INIT_PATH = "tcp://{}:12399".format(os.environ["SLURMD_NODENAME"])

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")

    # train cfg.RUN_N_TIMES times
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1
    if count > cfg.RUN_N_TIMES:
        raise ValueError(
            f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

    cfg.freeze()
    return cfg

def parsers_add(parser):
    parser.add_argument('--dataset', type=str, default='ADAM', help='dataset')
    parser.add_argument(
        '--bs', type=int, default=8, help='batch size for training set'
    )
    parser.add_argument('--num', type=int, default=0, help='few shot nums')
    parser.add_argument('--imbalance', type=int, default=1, help='balanced sampler')
    parser.add_argument('--data_dir', type=str, default='/mnt/data1/llr_data/AMD_Classification/', help='dataset')
    args = parser.parse_args()
    return args
    
def data(cfg):
    # 1. dataset
    if cfg.DATA.NAME == 'ARIA':
        preprocess = get_transforms('train', cfg.DATA.CROPSIZE)
        db_train_ARIA = DL.ARIADataSet(base_dir=cfg.DATA.DATAPATH, dataset='ARIA', split='Train', transform=preprocess, num_shot=cfg.DATA.num)
        preprocess = get_transforms('test', cfg.DATA.CROPSIZE)
        db_test_ARIA = DL.ARIADataSet(base_dir=cfg.DATA.DATAPATH, dataset='ARIA', split='Test', transform=preprocess)
        train_loader_ARIA = DataLoader(db_train_ARIA, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=1)
        test_loader_ARIA = DataLoader(db_test_ARIA, batch_size=2, shuffle=False, num_workers=1)
        train_gen = train_loader_ARIA
        test_gen = test_loader_ARIA
        valid_gen = train_loader_ARIA
        train_dataset = db_train_ARIA
        test_dataset = db_test_ARIA
        # valid_gen = v_loader_ARIA
    elif cfg.DATA.NAME  == 'STARE':
        preprocess = get_transforms('train', cfg.DATA.CROPSIZE)
        db_train_STARE = DL.STAREDataSet(base_dir=cfg.DATA.DATAPATH, dataset='STARE', split='Train', transform=preprocess, num_shot=cfg.DATA.num)
        preprocess = get_transforms('test', cfg.DATA.CROPSIZE)
        db_test_STARE = DL.STAREDataSet(base_dir=cfg.DATA.DATAPATH, dataset='STARE', split='Test', transform=preprocess)
        train_loader_STARE = DataLoader(db_train_STARE, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=1)
        test_loader_STARE = DataLoader(db_test_STARE, batch_size=2, shuffle=False, num_workers=1)
        train_gen = train_loader_STARE
        valid_gen = train_loader_STARE
        test_gen = test_loader_STARE
        train_dataset = db_train_STARE
        test_dataset = db_test_STARE
    elif cfg.DATA.NAME  == 'ADAM':
        preprocess = get_transforms('train', cfg.DATA.CROPSIZE)
        db_train_ADAM = DL.ADAMDataSet(base_dir=cfg.DATA.DATAPATH, dataset='ADAM', split='Train', transform=preprocess, num_shot=cfg.DATA.num)
        preprocess = get_transforms('test', cfg.DATA.CROPSIZE)
        db_test_ADAM = DL.ADAMDataSet(base_dir=cfg.DATA.DATAPATH, dataset='ADAM', split='Test', transform=preprocess)
        # db_v_ADAM = DL.ADAMDataSet(base_dir=cfg.DATA.DATAPATH, dataset='ADAM', split='Validation', transform=preprocess)
        train_loader_ADAM = DataLoader(db_train_ADAM, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=1)
        test_loader_ADAM = DataLoader(db_test_ADAM, batch_size=1, shuffle=False, num_workers=1)
        # v_loader_ADAM = DataLoader(db_v_ADAM, batch_size=2, shuffle=False, num_workers=1)
        train_gen = train_loader_ADAM
        test_gen = test_loader_ADAM
        # valid_gen = v_loader_ADAM
        valid_gen = None
        train_dataset = db_train_ADAM
        test_dataset = db_test_ADAM
    elif cfg.DATA.NAME  == 'ODIR':
        preprocess = get_transforms('train', cfg.DATA.CROPSIZE)
        db_train_ADAM = DL.ODIRDataSet(base_dir=cfg.DATA.DATAPATH, dataset='ODIR', split='Train', transform=preprocess, num_shot=cfg.DATA.num)
        preprocess = get_transforms('test', cfg.DATA.CROPSIZE)
        db_test_ADAM = DL.ODIRDataSet(base_dir=cfg.DATA.DATAPATH, dataset='ODIR', split='Test', transform=preprocess)
        # db_v_ADAM = DL.ADAMDataSet(base_dir=cfg.DATA.DATAPATH, dataset='ADAM', split='Validation', transform=preprocess)
        train_loader_ADAM = DataLoader(db_train_ADAM, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=1)
        test_loader_ADAM = DataLoader(db_test_ADAM, batch_size=2, shuffle=False, num_workers=1)
        # v_loader_ADAM = DataLoader(db_v_ADAM, batch_size=2, shuffle=False, num_workers=1)
        train_gen = train_loader_ADAM
        test_gen = test_loader_ADAM
        # valid_gen = v_loader_ADAM
        valid_gen = None
        train_dataset = db_train_ADAM
        test_dataset = db_test_ADAM
    
    return train_gen, test_gen, valid_gen, train_dataset, test_dataset


def get_loaders(cfg, logger):
    logger.info("Loading training data (final training data for vtab)...")
    train_loader, test_loader, val_loader, train_dataset, test_dataset = data(cfg)
    
    logger.info("Loading validation data...")
    # not really needed for vtab
    logger.info("Loading test data...")

    ## few shot
    label_list = []
    for i in train_dataset:
        label_list.append(i['label'])
    y_train = np.array(label_list)  # so, there are all the data samples here!!!!!!!

    print("*" * 100)
    print(f"Number of positive samples: {len([i for i in y_train if i == 0])}")
    print(f"Number of negative samples: {len([i for i in y_train if i == 1])}")
     
    # if cfg.DATA.num:
    #     print('--------------------few shot-------------------')
    #     if cfg.DATA.num == 1: # one-shot
    #         sampler = samplers.MPerClassSampler(y_train.flatten(), m=1, length_before_new_iter=cfg.DATA.num)
    #     else:  # others
    #         sampler = samplers.MPerClassSampler(y_train.flatten(), m=2, length_before_new_iter=cfg.DATA.num)
    #     train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler)
    #     val_loader = train_loader
    
    return train_loader, val_loader, test_loader


def train(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here

    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)
    print('------------------------', cfg.MODEL.TRANSFER_TYPE)
    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader)
    else:
        print("No train loader presented. Exit")

    if cfg.SOLVER.TOTAL_EPOCH == 0:
        trainer.eval_classifier(test_loader, "test", 0)


def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
