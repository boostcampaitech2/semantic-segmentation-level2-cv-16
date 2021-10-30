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
import yaml

from model import get_seg_model
from dataloader import *
from utils import *

# from loss import create_criterion
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb
import torch.nn as nn
import torch.nn.functional as F

# from loss import FocalLoss
import math
import segmentation_models_pytorch as smp
from tqdm import tqdm


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
    train_path = data_dir + "/train_v1.json"
    val_path = data_dir + "/valid_v1.json"
    seed_everything(args.seed)
    save_dir = "./" + increment_path(os.path.join(model_dir, args.name))
    os.makedirs(save_dir)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- augmentation
    train_transform = A.Compose(
        [
            # A.HorizontalFlip(p=0.5),
            # A.Rotate(limit=90),
           # A.Rotate(limit=90),
            A.RandomResizedCrop(512,512),
            # A.OneOf([
            #     # A.RandomCrop (128, 128),
            #     # A.RandomCrop (256, 256),
            #     A.Rotate(limit=90)
            # ]),
            # A.RandomCrop (256, 256),
            # A.OneOf([
            #     A.Resize(256,256),
            #     A.Resize(384,384),
            #     A.Resize(512,512),
            #     A.Resize(768,768),
            #     A.Resize(1024,1024)
            # ]),
            #A.RandomScale(p=1, scale_limit = [0.5, 2.0]),
            A.Resize(1024,1024),
            ToTensorV2()
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
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    # -- model
    config_path = '/opt/ml/segmentation/semantic-segmentation-level2-cv-16/dev/model_develop/HRNet/configs/hrnet_w64_seg_ocr.yaml'
    with open(config_path) as f:
        cfg = yaml.load(f)
    model = get_seg_model(cfg)
    model = model.to(device)
    config = wandb.config
    config.learning_rate = args.lr
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=1600, T_mult=2, eta_max=4e-5, T_up =800, gamma = 0.5)
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
    n_class = 11

    # -- logging
    best_mIoU = 0
    best_val_loss = 99999999
    for epoch in range(1, args.epochs + 1):
        # train loop
        model.train()
        loss_value = 0
        hist = np.zeros((11, 11))
        for idx, (images, masks, _) in enumerate(tqdm(train_loader)):
            wandb.log({"learning rate" : scheduler.get_lr()[0]})
            images = torch.stack(images)  # (batch, channel, height, width)
            masks = torch.stack(masks).long()
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            for i in range(len(outputs)):
                output = outputs[i]
                ph, pw = output.size(2), output.size(3)
                h, w = masks.size(1), masks.size(2)
                if ph != h or pw != w:
                    output = F.interpolate(input=output, size=(
                        h, w), mode='bilinear', align_corners=True)
                outputs[i] = output

            loss = 0
            for i in range(len(outputs)):
                loss += criterion(outputs[i], masks)
            outputs = outputs[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            loss_value += loss.item()

            if (idx + 1) % 75 == 0:
                train_loss = loss_value / 75
                print(
                    f"Epoch [{epoch}/{args.epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}"
                )
                wandb.log({"train/mIoU" : round(mIoU,4), "train/loss": train_loss})
                loss_value = 0
                wandb.log(
                    {
                        "train_image": wandb.Image(
                            images[0, :, :, :],
                            masks={
                                "predictions": {
                                    "mask_data": outputs[0, :, :],
                                    "class_labels": class_labels,
                                },
                                "ground_truth": {
                                    "mask_data": masks[0, :, :],
                                    "class_labels": class_labels,
                                },
                            },
                        )
                    }
                )

        hist = np.zeros((11, 11))

        ############validation##############

        with torch.no_grad():
            cnt = 0
            total_loss = 0
            print(f'Start validation #{epoch}')
            print("Calculating validation results...")
            model.eval()
            for idx, (images, masks, _) in enumerate(tqdm(val_loader)):
                images = torch.stack(images)  # (batch, channel, height, width)
                masks = torch.stack(masks).long()
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                for i in range(len(outputs)):
                    output = outputs[i]
                    ph, pw = output.size(2), output.size(3)
                    h, w = masks.size(1), masks.size(2)
                    if ph != h or pw != w:
                        output = F.interpolate(input=output, size=(
                            h, w), mode='bilinear', align_corners=True)
                    outputs[i] = output

                loss = 0
                for i in range(len(outputs)):
                    loss += criterion(outputs[i], masks)
                outputs = outputs[0]
                total_loss += loss
                cnt += 1

                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=11)

                if idx % args.vis_every == 0:
                    wandb.log(
                        {
                            "valid_image": wandb.Image(
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
                    f"New best model for val mIoU : {round(mIoU,4)}! saving the best model.."
                )
                torch.save(model.state_dict(), f"{save_dir}/{args.name}_HRNetV2_W64_OCR_{epoch}_{round(mIoU,4)}.pth")
                best_mIoU = mIoU
            if epoch == args.epochs:
                torch.save(model.state_dict(), f"{save_dir}/{args.name}_HRNetV2_W64_OCR_{epoch}_{round(mIoU,4)}.pth")
            print(
                f"Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}"
            )
            print(f"IoU by class : {IoU_by_class}")
        # val loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=16, help="random seed (default: 16)"
    )
    parser.add_argument(
        "--epochs", type=int, default=120, help="number of epochs to train (default: 25)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="input batch size for training (default: 2)",
    )
    # parser.add_argument('--model', type=str, default='Unet3plus', help='model type (default: DeepLabV3Plus)')
    parser.add_argument(
        "--lr", type=float, default=1e-7, help="learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--name", default="52_", help="model save at {SM_MODEL_DIR}/{name}"
    )
    parser.add_argument("--log_every", type=int, default=25, help="logging interval")
    parser.add_argument(
        "--vis_every", type=int, default=10, help="image logging interval"
    )

    # Container environment
    args = parser.parse_args()

    wandb.init(project="segmentation", entity="passion-ate")
    wandb.run.name = f"{args.name}HRNetV2_W64_OCR_cos_randomresizecrop_1024upscale"
    wandb.config.update(args)
    print(args)

    data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/segmentation/input/data")
    model_dir = os.environ.get("SM_MODEL_DIR", "./result")

    train(data_dir, model_dir, args)
