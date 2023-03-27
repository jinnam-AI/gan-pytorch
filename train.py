from tools.train import get_tools
from core.config import cfg
from runner.trainer import Trainer

class Train:
    def __init__(self) -> None:
        self.model = "lsgan512"
        self.image_size = 512
        self.ngf = 256
        self.ndf = 256
        self.max_epoch = 10000
        self.learning_rate = 0.0002
        self.batch_size = 32
        
    def _set_config(self):
        cfg.model = self.model
        cfg.image_size = self.image_size
        cfg.gen_feature = self.ngf
        cfg.dec_feature = self.ndf
        cfg.max_epoch = self.max_epoch
        cfg.learning_rate = self.learning_rate
        cfg.batch_size = self.batch_size
        cfg.transform_config["Resize"] = {"size":(self.image_size,self.image_size)}
        cfg.transform_config["CenterCrop"] = {"size":(self.image_size,self.image_size)}
        return cfg
    
    def run(self):
        cfg = self._set_config()
        tools = get_tools(cfg)
        trainer = Trainer(
            cfg=cfg,
            generator=tools[0],
            discriminator=tools[1],
            optimizerD=tools[2],
            optimizerG=tools[3],
            criterion=tools[4],
            logger=tools[5],
            device=tools[6]
            
        )
        trainer.train(tools[7])
        
if __name__=="__main__":
    train = Train()
    train.run()