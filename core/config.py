import pathlib

from typing import Dict,Any

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent

class Config:
    data_root:str = f"{PROJECT_DIR}/data"
    work_dir:str = f"{PROJECT_DIR}/result"
    exp_name:str = None
    
    model:str = "dcgan64"
    
    workers:int = 2
    batch_size:int = 64
    max_epoch:int = 5
    save_period:int = 100
    
    image_size:int = 64
    num_channel:int = 3
    
    vector:int = 100
    gen_feature:int = 64
    dec_feature:int = 64
    
    learning_rate:float = 0.0002
    beta1:float = 0.5
    
    show:bool = True
    
    transform_config:Dict[str,Any] = {
        "Resize":{"size":(128,128)},
        "CenterCrop":{"size":(128,128)},
        "Normalize":{"mean":(0.5,0.5,0.5),"std":(0.5,0.5,0.5)},
    }
    
cfg = Config()

