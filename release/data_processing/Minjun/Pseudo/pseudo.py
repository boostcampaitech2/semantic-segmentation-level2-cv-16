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

from smp_unet import *
from dataloader import *
from utils import *
from loss import *

# from loss import create_criterion
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb

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
            ##위의 2개만 사용했을때 현재 가장 좋은 성능을 보인다
            #A.ShiftScaleRotate(),
            #A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # data loader
    train_dataset = PseudoTrainset(
        data_dir=train_path, transform=train_transform
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
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    model = myModel("DeepLabV3Plus","timm-efficientnet-b4")
    model = model.to(device)
    wandb.watch(model)

    #criterion = JaccardLoss(mode='multiclass', classes=11, smooth=0.0)
    criterion = [nn.CrossEntropyLoss(), DiceLoss(mode='multiclass', classes=11, smooth=0.0)]
    optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=1e-3)

    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
    class_labels = {
        0: "Backgroud",
        1: "General trash",
        2: "Paper",
        3: "Paper pack",
        4: "Metal",
        5: "Glass",
        6: "Plastic",
        7: "Styrofoam",
        8: "Plastic bag",
        9: "Battery",
        10: "Clothing",
    }
    # -- logging
    best_mIoU = 0
    best_val_loss = 99999999
    for epoch in range(1, args.epochs + 1):
        # train loop
        model.train()
        loss_value = 0
        for idx, (images, masks) in enumerate(train_loader):
            loss = 0
            images = torch.stack(images)  # (batch, channel, height, width)
            masks = torch.stack(masks).long()
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            optimizer.zero_grad()

            for i in criterion:
                loss += i(outputs, masks) 
            #loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            loss_value += loss.item()

            if (idx + 1) % 25 == 0:
                train_loss = loss_value / 25
                print(
                    f"Epoch [{epoch}/{args.epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {round(loss.item(),4)}"
                )
                wandb.log({"train/loss": train_loss})
                loss_value = 0
        hist = np.zeros((11, 11))

        ############validation##############

        with torch.no_grad():
            cnt = 0
            total_loss = 0
            print("Calculating validation results...")
            model.eval()
            for idx, (images, masks, _) in enumerate(val_loader):
                loss = 0
                images = torch.stack(images)  # (batch, channel, height, width)
                masks = torch.stack(masks).long()
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                
                for j in criterion:
                    loss += j(outputs, masks)
                #loss = criterion(outputs, masks)
                total_loss += loss
                cnt += 1

                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=11)

                if idx % args.vis_every == 0:
                    wandb.log(
                        {
                            "visualize": wandb.Image(
                                images[0, :, :, :],
                                masks={
                                    "predictions": {
                                        "mask_data": outputs[0, :, :],
                                        "class_labels": class_labels,
                                    },
                                    "ground_truth": {
                                        "mask_data": masks[0, :, :]
                                        .detach()
                                        .cpu()
                                        .numpy(),
                                        "class_labels": class_labels,
                                    },
                                },
                            )
                        }
                    )
            # val_loss = np.sum(val_loss_items) / len(val_loader)
            # best_val_loss = min(best_val_loss, val_loss)
            acc, _, mIoU, _, IoU = label_accuracy_score(hist)
            IoU_by_class = [
                {classes: round(IoU, 4)} for IoU, classes in zip(IoU, category_names)
            ]
            avrg_loss = total_loss / cnt
            best_val_loss = min(avrg_loss, best_val_loss)

            log = {
                "val/mIoU": mIoU,
                "val/loss": avrg_loss,
                "val/accuracy": acc,
            }
            for d in IoU_by_class:
                for cls in d:
                    log[f"val/{cls}_IoU"] = d[cls]
            wandb.log(log)
            if mIoU > best_mIoU:
                print(
                    f"New best model for val mIoU : {mIoU:4.2%}! saving the best model.."
                )
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                best_mIoU = mIoU
            torch.save(model.state_dict(), f"{save_dir}/last.pth")
            print(
                f"Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}"
            )
            print(f"IoU by class : {IoU_by_class}")
        scheduler.step()
        # val loop


if __name__ == "__main__":
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
        "--lr", type=float, default=1e-5, help="learning rate (default: 5e-6)"
    )
    parser.add_argument(
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )
    parser.add_argument("--log_every", type=int, default=25, help="logging interval")
    parser.add_argument(
        "--vis_every", type=int, default=10, help="image logging interval"
    )

    # Container environment
    args = parser.parse_args()

    wandb.init(project="segmentation", entity="passion-ate")
    wandb.run.name = "module test"
    wandb.config.update(args)
    print(args)

    data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/segmentation/input/data")
    model_dir = os.environ.get("SM_MODEL_DIR", "./test")

    train(data_dir, model_dir, args)