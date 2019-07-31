import torch
import torchvision

import pandas as pd
import numpy as np

from PIL import Image


def blend(origin, mask1=None, mask2=None):
    img = torchvision.transforms.functional.to_pil_image(origin)
    if mask1 is not None:
        mask1 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.zeros(1, *mask1.size()),
            torch.stack([mask1.float()]),
            torch.zeros(1, *mask1.size()),
        ]))
        img = Image.blend(img, mask1, 0.5)
        
    if mask2 is not None:
        mask2 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.stack([mask2.float()]),
            torch.zeros(1, *mask2.size()),
            torch.zeros(1, *mask2.size()),
        ]))
        img = Image.blend(img, mask2, 0.5)
    
    return img