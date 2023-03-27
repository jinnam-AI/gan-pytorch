import torch
import torch.nn as nn
from torch import Tensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class LSloss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self,input:Tensor,targer:Tensor):
        return  0.5 * torch.mean((input-targer)**2)

class BeGanloss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self,input:Tensor,targer:Tensor):
        return  torch.mean(torch.abs(input-targer)**2)