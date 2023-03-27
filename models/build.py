from .dcgan import (generator64,generator128,generator256,generator512,
                    discriminator64,discriminator128,discriminator256)
from .lsgan import lsdiscriminator64,lsdiscriminator128,lsdiscriminator256,lsdiscriminator512

MODELS = {
    "dcgan64":[generator64,discriminator64],
    "dcgan128":[generator128,discriminator128],
    "dcgan256":[generator256,discriminator256],
    "lsgan64":[generator64,lsdiscriminator64],
    "lsgan128":[generator128,lsdiscriminator128],
    "lsgan256":[generator256,lsdiscriminator256],
    "lsgan512":[generator512,lsdiscriminator512],
}

def get_model(model:str,vector_size:int=100,ngf:int=64,ndf:int=64):
    models = MODELS.get(model,None)
    
    if models is None:
        raise ModuleNotFoundError(f"해당하는 모델이 존재하지 않습니다. 모델 : {model}")
    
    generator = models[0](vector_size,ngf)
    discriminator = models[1](ndf)
    return generator,discriminator