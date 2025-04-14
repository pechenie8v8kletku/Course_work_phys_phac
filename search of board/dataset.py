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
        image = np.array(Image.open(img_path).convert("RGB"))
        with open(mask_path, 'r') as f:
            data = json.load(f)
        mask=maskrle.decode(data['segmentation'])
        mask = torch.tensor(mask)
        mask = mask.to(torch.uint8)
        mask_padded = mask.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.uint8)
        output = F.conv2d(mask_padded, kernel, padding=1)
        output = output.squeeze(0).squeeze(0)
        mask=((output>0).float()).numpy()

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

