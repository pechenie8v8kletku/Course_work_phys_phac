import torch
import os
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from model import UNET
from Connected_regions import find_connected_regions
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIR = "for_test"
OUTPUT_DIR = "after_test"
INPUT_DIR2 = "for_test2"
HEIGHT=512
WIDTH=512


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])
def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    model.eval()

    def trans(img):
        val_transforms = A.Compose(
            [
                A.Resize(height=HEIGHT, width=WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
        val_transforms_iz = A.Compose(
            [
                A.Resize(height=HEIGHT, width=WIDTH),
            ],
        )
        img = np.array(img)
        img_tensor = val_transforms(image=img)["image"].unsqueeze(0)
        trans = val_transforms_iz(image=img)["image"]
        img_tensor = img_tensor.to(device=DEVICE)
        torch.set_printoptions(precision=40, threshold=300, edgeitems=100)
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.sigmoid(output).squeeze(0)
            pred = (pred > 0.99).float()
        return pred.cpu().numpy().reshape(512, 512), np.asarray(trans)

    images = os.listdir(INPUT_DIR)
    images_dif = os.listdir(INPUT_DIR2)
    for i in range(len(images)):
        img = os.path.join(INPUT_DIR, images[i])
        img_dif = os.path.join(INPUT_DIR2, images_dif[i])
        img = Image.open(img).convert("HSV")
        img_dif = Image.open(img_dif).convert("HSV")
        pred_image, transform_image = trans(img)
        pred_dif, transform_image_dif = trans(img_dif)
        hsv_mask = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float16)
        coefs1 = (np.sum(transform_image, axis=(0, 1))) / (HEIGHT * WIDTH)
        coefs2 = (np.sum(transform_image_dif, axis=(0, 1))) / (HEIGHT * WIDTH)
        orthogonal = (1, coefs2[1] / coefs1[1], coefs2[2] / coefs1[2])
        transform_image = transform_image * orthogonal


        segments= find_connected_regions(pred_image, 150,130)
        segments_dif=find_connected_regions(pred_dif,150,130)
        for segment in segments:
            segment = segment[:, :, np.newaxis]
            h1,s1,v1=(np.sum(np.asarray(segment)*transform_image,axis=(0,1))/np.sum(segment))[:, np.newaxis]
            for segment_dif in segments_dif:
                segment_dif = segment_dif[:, :, np.newaxis]
                intersection=segment*segment_dif
                size = np.sum(intersection)

                if size > 50:
                    overlap = size/(np.sum(segment)+np.sum(segment_dif)-size)
                    h2, s2, v2 =(np.sum(np.asarray(segment_dif)*transform_image_dif,axis=(0,1))/np.sum(segment_dif))[:, np.newaxis]
                    delta_h = abs(h1-h2)
                    delta_s = abs(s2-s1)
                    delta_v = abs(v1-v2)
                    area=intersection+overlap*(segment-intersection)
                    hsv_mask =hsv_mask+ np.stack((delta_h*area,delta_s*area,delta_v*area),axis=2).squeeze()
        print(np.max(hsv_mask, axis=(0, 1)))
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        img = Image.fromarray(hsv_mask.astype(np.uint8), mode="HSV").convert("RGB")
        img.save(os.path.join(OUTPUT_DIR, f"{i}.png"))

if __name__ == "__main__":
    main()