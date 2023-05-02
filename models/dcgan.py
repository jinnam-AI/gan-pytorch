import math
import torch.nn as nn

from .utils import weights_init

class Gblcok(nn.Module):
    def __init__(self,
                 input_size:int,
                 output_size:int,
                 stride:int=1,
                 padding:int=0) -> None:
        super(Gblcok,self).__init__()
        self.convt2d = nn.ConvTranspose2d(input_size,output_size,
                                          kernel_size=4,stride=stride,padding=padding,bias=False)
        self.batchnorm = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU(True)
    
    def forward(self,x):
        x = self.convt2d(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class Generator(nn.Module):
    def __init__(self,
                 vector_size:int=100,
                 image_size:int=64,
                 feeature:int=64) -> None:
        super(Generator,self).__init__()
        self.image_size = image_size
        self.init_output = feeature*8
        self.init_block = Gblcok(vector_size,self.init_output,1,0)
        self.layers = self._make_layer()
        
    def _make_layer(self):
        b_nums = int(math.log(self.image_size/4,2)-1)
        layers = []
        for i in range(b_nums):
            layers.append(
                Gblcok(int(self.init_output/2**i),int(self.init_output/(2**(i+1))),2,1)
            )
        layers.append(nn.ConvTranspose2d(int(self.init_output/(2**(i+1))),3,kernel_size=4,stride=2,padding=1,bias=False))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.init_block(x)
        x = self.layers(x)
        return x
    
class Dblock(nn.Module):
    def __init__(self,
                 input_size:int,
                 output_size:int,
                 stride:int,
                 padding:int) -> None:
        super(Dblock,self).__init__()
        self.conv2d = nn.Conv2d(input_size, output_size, 4, stride, padding, bias=False)
        self.batchnrom = nn.BatchNorm2d(output_size)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self,x):
        x = self.conv2d(x)
        x = self.batchnrom(x)
        x = self.leakyrelu(x)
        return x
            
class Discriminator(nn.Module):
    def __init__(self,image_size:int=64,feeature:int=64) -> None:
        super(Discriminator,self).__init__()
        self.image_size = image_size
        self.init_output = int(feeature*8/2**(math.log2(image_size/4)-1))
        self.init_block = self._init_block()
        self.layers = self._make_layer()
        
    def _init_block(self):
        return nn.Sequential(
            nn.Conv2d(3,self.init_output,4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def _make_layer(self):
        b_nums = int(math.log(self.image_size/4,2)-1)
        layers = []
        for i in range(b_nums):
            layers.append(
                Dblock(int(self.init_output*(2**i)),int(self.init_output*(2**(i+1))),2,1)
            )
        layers.append(nn.Conv2d(int(self.init_output*(2**(i+1))),1,kernel_size=4,stride=1,padding=0,bias=False))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.init_block(x)
        x = self.layers(x)
        return x
    
def generator64(vector_size:int=100,feature:int=64):
    generator = Generator(vector_size,64,feature)
    generator.apply(weights_init)
    return generator

def generator128(vector_size:int=100,feature:int=64):
    generator = Generator(vector_size,128,feature)
    generator.apply(weights_init)
    return generator

def generator256(vector_size:int=100,feature:int=64):
    generator = Generator(vector_size,256,feature)
    generator.apply(weights_init)
    return generator

def discriminator64(feature:int=64):
    discriminator = Discriminator(64,feature)
    discriminator.apply(weights_init)
    return discriminator

def discriminator128(feature:int=64):
    discriminator = Discriminator(128,feature)
    discriminator.apply(weights_init)
    return discriminator

def discriminator256(feature:int=64):
    discriminator = Discriminator(256,feature)
    discriminator.apply(weights_init)
    return discriminator