import os
import re
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader

from core.config import Config
from models.build import get_model
from models.utils import LSloss
from loguru import logger
from utils.transforms import get_transforms


def get_tools(cfg:Config):
    #exp_name
    if cfg.exp_name is None:
        cfg.exp_name = cfg.model
    
    #log 
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_path = f"{cfg.work_dir}/{cfg.exp_name}/log"
    os.makedirs(log_path, exist_ok=True)
    log_file =  f'{log_path}/{timestamp}_train.log'
    logger.add(log_file)
    
    #work_dir
    setattr(cfg,"exp_work_dir",f"{cfg.work_dir}/{cfg.exp_name}")
    
    #data
    transforms = get_transforms(cfg.transform_config)
    logger.info(transforms)
    
    dataset = ImageFolder(cfg.data_root,transform=transforms)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers
        )
    
    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #criterion
    if re.match(r"lsgan",cfg.model):
        criterion = LSloss()
    else:
        criterion = nn.BCELoss()
    logger.info(f"criterion:{str(criterion)}")
    
    #model
    generator,discriminator = get_model(cfg.model,cfg.vector,cfg.gen_feature,cfg.dec_feature)
    if cfg.restart:
        generator.load_state_dict(torch.load(cfg.checkpoint_gen))
        discriminator.load_state_dict(torch.load(cfg.checkpoint_dec))
    
    generator.to(device)
    discriminator.to(device)
    
    logger.info(generator)
    logger.info(discriminator)
    
    optimizerD = optim.Adam(discriminator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
    return generator,discriminator,optimizerD,optimizerG,criterion,logger,device,dataloader

