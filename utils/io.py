import os
import shutil

def init_dir(path:str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path,exist_ok=True)