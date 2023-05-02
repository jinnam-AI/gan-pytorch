import torch
from models.build import generator64

from torchvision.utils import make_grid,save_image

import torch.multiprocessing as mp
from torch.multiprocessing import spawn

from utils.io import init_dir


class Generation:
    def __init__(self) -> None:
        self.save_path
        self.checkpoint
        self.img_num
        

    def generate(save_path:str,checkpoint:str,img_num:int):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        generator = generator64()
        generator.load_state_dict(torch.load(checkpoint))
        generator.eval()
        generator.to(device)
        
        for i in range(img_num):
            noise = torch.randn((1,100,1,1),device=device)
            fake = generator(noise)
            fake = fake.detach().cpu()
            image = make_grid(fake)
            
            img_file = f"{save_path}/{i}.png"
            save_image(image,img_file)
        
if __name__=="__main__":
    train = Generation()
    train.run()