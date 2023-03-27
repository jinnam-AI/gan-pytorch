import torchvision.transforms.transforms as tf

from typing import Dict,Any

transforms_scope = dict(
    Resize=tf.Resize,
    CenterCrop=tf.CenterCrop,
    Normalize=tf.Normalize
)

def get_transforms(cfg:Dict[str,Any]):
    tf_list = []
    for key,value in cfg.items():
        transform = transforms_scope[key](**value)
        if key == "Normalize":
            tf_list.append(tf.ToTensor())
        tf_list.append(transform)
    transforms = tf.Compose(tf_list)
    return transforms