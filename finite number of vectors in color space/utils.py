import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np
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
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

## переделать хуету снизу##
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)


            dice_coeff = 1

    print(f"Dice score: {dice_coeff/len(loader)}")
    model.train()


def hsv_to_rgb(h, s, v):
    return tuple(int(c * 255) for c in Image.Color.getrgb(f"hsv({h},{s}%,{v}%)"))
def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        torch.set_printoptions(precision=40, threshold=300, edgeitems=100)
        x = x.to(device=device)
        with torch.no_grad():
            y_pred = F.softmax(model(x), dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()

        height, width = y_pred.shape[1], y_pred.shape[2]
        hsv_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_idx, hsv in class_to_color .items():
            hsv_mask[y_pred[0] == class_idx] = hsv

        img = Image.fromarray(hsv_mask, mode="RGB")
        img.save(f"{folder}pred_{idx}.png")
    model.train()
