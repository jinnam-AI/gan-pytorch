from tools.train import get_tools
from core.config import cfg
from runner.dc_trainer import DcTrainer

class Train:
    def __init__(self) -> None:
        self.model = "lsgan128"
        self.image_size = 128
        self.ngf = 128
        self.ndf = 128
        self.max_epoch = 10000
        self.learning_rate = 0.001
        self.batch_size = 16
        self.restart = False
        self.start_epoch = 1
        self.checkpoint_gen = "D:/Workplace/AI/gan/result/lsgan64/weights/epoch_3900.pth"
        self.checkpoint_dec = "D:/Workplace/AI/gan/result/lsgan64/weights/latest_d.pth"
        
    def _set_config(self):
        cfg.model = self.model
        cfg.image_size = self.image_size
        cfg.gen_feature = self.ngf
        cfg.dec_feature = self.ndf
        cfg.max_epoch = self.max_epoch
        cfg.learning_rate = self.learning_rate
        cfg.batch_size = self.batch_size
        
        if self.restart:
            cfg.restart = self.restart
            cfg.start_epoch = self.start_epoch
            cfg.checkpoint_gen = self.checkpoint_gen
            cfg.checkpoint_dec = self.checkpoint_dec
            
        cfg.transform_config["Resize"] = {"size":(self.image_size,self.image_size)}
        cfg.transform_config["CenterCrop"] = {"size":(self.image_size,self.image_size)}
        return cfg
    
    def run(self):
        cfg = self._set_config()
        tools = get_tools(cfg)
        trainer = DcTrainer(
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