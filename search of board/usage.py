import torch
import os
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from model import UNET
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIR="for_usage"
OUTPUT_DIR="after_model"
def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])
def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    model.eval()
    images=os.listdir(INPUT_DIR)
    for i in range(len(images)):
        img=os.path.join(INPUT_DIR, images[i])
        img = Image.open(img).convert("HSV")
        val_transforms = A.Compose(
            [
                A.Resize(height=512, width=512),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
        img = np.array(img)
        img_tensor = val_transforms(image=img)["image"].unsqueeze(0)
        img_tensor = img_tensor.to(device=DEVICE)
        torch.set_printoptions(precision=40, threshold=300, edgeitems=100)
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.sigmoid(output).squeeze(0)
            pred = (pred < 0.97).float()


        pred_image = pred.cpu().numpy().reshape(512, 512)
        height, width = pred_image.shape[0], pred_image.shape[1]
        hsv_mask=(pred_image*255).astype(np.uint8)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        img = Image.fromarray(hsv_mask,mode='L')
        img.save(os.path.join(OUTPUT_DIR, f"{i}.png"))

if __name__ == "__main__":
    main()