import torch
from typing import Tuple

from torchvision.datasets import ImageFolder

class MedDataset(ImageFolder):
    
    def __init__(self, root: str, transform: torch.nn.Module = None):
        super().__init__(root, transform)
        
        self.name = root.split("/")[-2]
        self.split = root.split("/")[-1]
        print(f"Dataset {self.name} split {self.split} loaded")
    
    def __getitem__(self, index: int)-> Tuple[torch.Tensor, int, str]:
        img_tensor, label = super().__getitem__(index)
        
        return img_tensor, label, self.imgs[index][0]
    
    def __getitem__from_name__(self, image_name: str) -> Tuple[torch.Tensor, int, str]:
        for index, img in enumerate(self.imgs):
            if image_name in img[0] :
                return self.__getitem__(index)
        return None