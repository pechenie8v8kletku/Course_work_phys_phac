import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pycocotools.mask as maskrle
import json
import torch.nn.functional as F

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.skelets=os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.skelets[index])
        image = np.array(Image.open(img_path).convert("HSV"))
        with open(mask_path, 'r') as f:
            data = json.load(f)
        result_mask = torch.zeros(data[0]["segmentation"]["size"])
        for aug in data:
            result_mask = result_mask + (torch.tensor(maskrle.decode(aug["segmentation"]))) * (aug["segmentation"][
                "class"]-1)
        result_mask=np.array(result_mask)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=result_mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask



