import torch
import os
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import torch.nn.functional as F
from model import ResUNet
from UNET_model import UNET
from torchvision import transforms
color_list={(0,20,230):1,
            (0,20,190):2,
            (0,20,140):3,
            (0,20,90):4,
            (0,20,20):5,
            (0, 220, 220): 6,
            (20, 220, 220): 9,
            (40, 220, 220): 12,
            (60, 220, 220): 15,
            (80, 220, 220): 18,
            (100, 220, 220): 21,
            (120, 220, 220): 24,
            (140, 220, 220): 27,
            (160, 220, 220): 30,
            (180, 220, 220): 6,
            (0, 90, 90): 8,
            (0, 150, 150): 7,
            (20, 150, 150): 10,
            (20, 90, 90): 11,
            (40, 150, 150): 12,
            (40, 90, 90): 13,
            (60, 150, 150): 16,
            (60, 90, 90): 17,
            (80, 150, 150): 19,
            (80, 90, 90): 20,
            (100, 150, 150): 22,
            (100, 90, 90): 23,
            (120, 150, 150): 25,
            (120, 90, 90): 26,
            (140, 150, 150): 28,
            (140, 90, 90): 29,
            (160, 150, 150): 31,
            (160, 90, 90): 32,
            (180, 150, 150): 7, (180, 90, 90): 8
             }
class_to_color = {v: k for k, v in color_list.items()}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_VECTORS=32
INPUT_DIR="for_usage"
OUTPUT_DIR="after_model"
def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])
def main():
    model = UNET(in_channels=3, out_channels=NUM_VECTORS).to(DEVICE)
    load_checkpoint(torch.load("my_checkpointUSAGE.pth.tar"), model)
    model.eval()
    images=os.listdir(INPUT_DIR)
    for i in range(len(images)):
        img=os.path.join(INPUT_DIR, images[i])
        img = Image.open(img).convert("HSV")
        val_transforms = A.Compose(
            [
                A.Resize(height=256, width=256),
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
            print("Output shape (logits):", output.shape)
            print("Output min, max (logits):", output.min().item(), output.max().item())
            output = F.softmax(output,dim=1).squeeze(0)
            print(output.shape)
            print("Output min, max (softmax):", output.min().item(), output.max().item())
            pred = torch.argmax(output, dim=0)
            print("Prediction shape:", pred.shape)


        pred_image = pred.cpu().numpy()
        height, width = pred_image.shape[0], pred_image.shape[1]
        hsv_mask = np.zeros((256, 256, 3), dtype=np.uint8)


        for class_idx, hsv in class_to_color .items():
            h,s,v=hsv
            hsv=h*255/180,s,v
            hsv_mask[pred_image == class_idx] = hsv
            print(hsv)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        img = Image.fromarray(hsv_mask, mode="HSV").convert("RGB")
        img.save(os.path.join(OUTPUT_DIR, f"{images[i]}color.png"))

if __name__ == "__main__":
    main()