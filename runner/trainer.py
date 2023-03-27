import os

import torch
import torch.nn as nn

from logging import Logger

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid,save_image

from core.config import Config
from loguru import logger

class Trainer:
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
        self.real_label = 1.
        self.fake_label = 0.
        self.show = cfg.show
        self.save_period = cfg.save_period
        self.max_epoch = cfg.max_epoch
        self.vector_size = cfg.vector
        self.batch_size = cfg.batch_size
    
    def _show_result(self,fake:torch.Tensor,imagefile:str):
        image = make_grid(fake)
        save_image(image,imagefile)
        
    def save_checkpoint(self,epoch:int):
        checkpoint_path = f"{self.work_dir}/weights"
        os.makedirs(checkpoint_path,exist_ok=True)
        
        checkpoint = f"{checkpoint_path}/epoch_{epoch}.pth"
        checkpoint_d = f"{checkpoint_path}/latest_d.pth"
        
        torch.save(self.generator.state_dict(),checkpoint)
        torch.save(self.discriminator.state_dict(),checkpoint_d)
        
    def _train_discriminator(self,data):
        self.discriminator.zero_grad()
            
        #real data training
        real_cpu = data[0].to(self.device)
        self.batch_size = real_cpu.size(0)
        self.label = torch.full((self.batch_size,),self.real_label,dtype=torch.float,device=self.device)
        
        output = self.discriminator(real_cpu).view(-1)
        
        errD_real = self.criterion(output,self.label)
        
        errD_real.backward()
        D_x = output.mean().item()
        
        # fake data training
        noise = torch.randn(self.batch_size,self.vector_size,1,1,device=self.device)
        
        fake = self.generator(noise)
        self.label.fill_(self.fake_label)
        output = self.discriminator(fake.detach()).view(-1)
        
        errD_fake = self.criterion(output,self.label)
        errD_fake.backward()
        
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        
        self.optimizerD.step()
        return fake,errD,D_G_z1,D_x
    
    def _train_generator(self,fake):
        self.generator.zero_grad()
        self.label.fill_(self.real_label)
        
        output = self.discriminator(fake).view(-1)
        
        errG = self.criterion(output,self.label)
        
        errG.backward()
        D_G_z2 = output.mean().item()
        
        self.optimizerG.step()
        return errG,D_G_z2
    
    def _inner_train(self,data_loader:DataLoader,epoch:int):
        for _iter,data in enumerate(data_loader):
            fake,errD,D_G_z1,D_x = self._train_discriminator(data)
            errG,D_G_z2 = self._train_generator(fake)
            
            if _iter%50 == 0 :
                progress = f"[{epoch:5d}/{self.max_epoch:5d}] [{_iter:5d}/{len(data_loader):5d}]"
                loss = f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)):{D_G_z1:.4f}/{D_G_z2:.4f}"
                self.logger.info(progress+" "+loss)
                
    @logger.catch
    def train(self,data_loader:DataLoader):
        for epoch in range(1,self.max_epoch):
            self._inner_train(data_loader,epoch)
            
            if epoch%self.save_period == 0:
                self.save_checkpoint(epoch)
            
            if self.show and epoch%self.save_period == 0:
                with torch.no_grad():
                    noise = torch.randn(32,self.vector_size,1,1,device=self.device)
                    fake = self.generator(noise).detach().cpu()
                    
                imagefile = f"{self.work_dir}/weights/epoch_{epoch}.png"
                self._show_result(fake,imagefile)