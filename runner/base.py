import os

import torch
import torch.nn as nn

from logging import Logger

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid,save_image

from core.config import Config
from loguru import logger


class BaseTrainer:
    def __init__(self,
                 cfg:Config,
                 generator:nn.Module,
                 discriminator:nn.Module,
                 optimizerG:Optimizer,
                 optimizerD:Optimizer,
                 criterion:nn.Module,
                 logger:Logger,
                 device:str="cuda:0",
                 ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.work_dir = cfg.exp_work_dir
        self.show = cfg.show
        self.save_period = cfg.save_period
        self.max_epoch = cfg.max_epoch
        self.vector_size = cfg.vector
        self.batch_size = cfg.batch_size
    
    def show_result(self,fake:torch.Tensor,imagefile:str):
        image = make_grid(fake)
        save_image(image,imagefile)
        
    def save_checkpoint(self,epoch:int):
        checkpoint_path = f"{self.work_dir}/weights"
        os.makedirs(checkpoint_path,exist_ok=True)
        
        checkpoint = f"{checkpoint_path}/epoch_{epoch}.pth"
        
        torch.save(self.generator.state_dict(),checkpoint)
        
    def _train_discriminator(self,data:torch.Tensor):
        pass
    
    def _train_generator(self,fake:torch.Tensor):
        pass
    
    def _inner_train(self,data_loader:DataLoader,epoch:int):
        pass
                
    @logger.catch
    def train(self,data_loader:DataLoader):
        pass