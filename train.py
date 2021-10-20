import argparse
import glob
import os


import random
import re
from importlib import import_module
from pathlib import Path
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from adamp import AdamP
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from model import *
from dataloader import *
from utils import *
from loss import create_criterion
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb
from loss import FocalLoss
import math
import torch.optim.lr_scheduler as lr_scheduler


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def increment_path(path, exist_ok=False):
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def collate_fn(batch):
    return tuple(zip(*batch))


def train(data_dir, model_dir, args):
    torch.backends.cudnn.benchmark = True
    train_path = data_dir + "/train.json"
    val_path = data_dir + "/val.json"
    seed_everything(args.seed)
    save_dir = "./" + increment_path(os.path.join(model_dir, args.name))
    os.makedirs(save_dir)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- augmentation
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(512, 512, (0.75, 1.0), p=0.5),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # data loader
    train_dataset = CustomDataLoader(
        data_dir=train_path, mode="train", transform=train_transform
    )
    val_dataset = CustomDataLoader(
        data_dir=val_path, mode="val", transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    # -- model
    model = UNet(n_classes=11)
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=1e-3)

    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)

    # -- logging
    best_mIoU = 0
    best_val_loss = 99999999
    for epoch in range(1, args.epochs + 1):
        # train loop
        model.train()
        loss_value = 0
        for idx, (images, masks, _) in enumerate(tqdm(train_loader)):
            images = torch.stack(images)  # (batch, channel, height, width)
            masks = torch.stack(masks).long()
            images, masks = images.to(device), masks.to(device)
            model = model.to(device)

            outputs = model(images)
            optimizer.zero_grad()

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            loss_value += loss.item()

            if (idx + 1) % 25 == 0:
                train_loss = loss_value / 25
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4}"
                )
                wandb.log({"train loss": train_loss})
                loss_value = 0
        hist = np.zeros((11, 11))

        ############validation##############

        with torch.no_grad():
            cnt = 0
            total_loss = 0
            print("Calculating validation results...")
            model.eval()
            for idx, (images, masks, _) in enumerate(val_loader):
                images = torch.stack(images)  # (batch, channel, height, width)
                masks = torch.stack(masks).long()
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss
                cnt += 1

                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=11)

            # val_loss = np.sum(val_loss_items) / len(val_loader)
            # best_val_loss = min(best_val_loss, val_loss)
            _, _, mIoU, _ = label_accuracy_score(hist)
            avrg_loss = total_loss / cnt
            best_val_loss = min(avrg_loss, best_val_loss)

            wandb.log({"Test mIoU": mIoU, "Test Loss": avrg_loss})
            if mIoU > best_mIoU:
                print(
                    f"New best model for val accuracy : {mIoU:4.2%}! saving the best model.."
                )
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                best_mIoU = mIoU
            torch.save(model.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] mIoU : {mIoU:4.2%}, loss: {avrg_loss:4.2} || "
                f"best mIoU : {best_mIoU:4.2%}, best loss: {best_val_loss:4.2}"
            )
        scheduler.step()
        # val loop


if __name__ == "__main__":
    wandb.init(project="segmentation")
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="number of epochs to train (default: 28)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )
    # parser.add_argument('--model', type=str, default='Unet3plus', help='model type (default: DeepLabV3Plus)')
    parser.add_argument(
        "--lr", type=float, default=5e-6, help="learning rate (default: 5e-6)"
    )
    parser.add_argument(
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )

    # Container environment
    args = parser.parse_args()

    wandb.run.name = "unet3plus"
    wandb.config.update(args)
    print(args)

    data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/segmentation/input/data")
    model_dir = os.environ.get("SM_MODEL_DIR", "./unet3plus_exp")

    train(data_dir, model_dir, args)