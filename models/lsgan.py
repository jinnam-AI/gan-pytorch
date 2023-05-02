import math
import torch.nn as nn
from .dcgan import Discriminator,Dblock,weights_init

class LSDiscriminator(Discriminator):
    def _make_layer(self):
        b_nums = int(math.log(self.image_size/4,2)-1)
        layers = []
        for i in range(b_nums):
            layers.append(
                Dblock(int(self.init_output*2**i),int(self.init_output*(2**(i+1))),2,1)
            )
        layers.append(nn.Conv2d(int(self.init_output*(2**(i+1))),1,kernel_size=4,stride=1,padding=0,bias=False))
        return nn.Sequential(*layers)

def lsdiscriminator64(feature:int=64):
    discriminator = LSDiscriminator(64,feature)
    discriminator.apply(weights_init)
    return discriminator

def lsdiscriminator128(feature:int=64):
    discriminator = LSDiscriminator(128,feature)
    discriminator.apply(weights_init)
    return discriminator

def lsdiscriminator256(feature:int=64):
    discriminator = LSDiscriminator(256,feature)
    discriminator.apply(weights_init)
    return discriminator