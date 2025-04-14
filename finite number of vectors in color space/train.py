import torch
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from monai.losses import DiceLoss
from UNET_model import UNET
import torch.nn.functional as F
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
ALPHA=0.997
ST_SIZE=5
GAMMA_LR=0.1
LEARNING_RATE = (1e-3)
WEIGHT_DECAY=(1e-2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 400
NUM_WORKERS = 2
NUM_VECTORS=32
IMAGE_HEIGHT = 256# 1280 originally
IMAGE_WIDTH = 256 # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "training_img"
TRAIN_MASK_DIR = "training_masks"
VAL_IMG_DIR = "valid_img"
VAL_MASK_DIR = "valid_masks"

'''class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        intersection = torch.sum(preds * targets_one_hot, dim=(2, 3))
        union = torch.sum(preds, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
'''
def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    running_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        targets_one_hot = F.one_hot(targets, num_classes=NUM_VECTORS).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        # forward
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets_one_hot)

        # backward
        optimizer.zero_grad()
        #print(f"Before scaler: {loss.item()}")
        #print(f"Does loss require grad? {loss.requires_grad}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #print(f"After scaler: {loss.item()}")
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        running_loss += loss.item()
    avg_loss = running_loss / num_batches
    return avg_loss


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=NUM_VECTORS).to(DEVICE)
    loss_fn = DiceLoss(softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=0.01,
        cooldown=2,
        min_lr=(1e-3)/2,
        verbose=True
    )'''
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-3/2,
        max_lr=(1e-3)*6,
        step_size_up=4,
        mode="triangular2"
    )

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):

        # save model
        if epoch%1==0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE
            )
            check_accuracy(val_loader, model, device=DEVICE)

        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print(avg_loss)
        scheduler.step(avg_loss)


if __name__ == "__main__":
    main()