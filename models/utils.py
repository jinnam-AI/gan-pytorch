import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def l2norm(tensor:Tensor):
    """
    l2 norm  : 지수가 2인 정규화

    Args:
        tensor (Tensor): 입력 tensor

    Returns:
        Tensor: 정규화된 tensor
    """
    return F.normalize(tensor, p = 2, dim = -1)

class LSloss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self,input:Tensor,target:Tensor):
        return  0.5 * torch.mean((input-target)**2)
    

    
    