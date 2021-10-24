#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import random

import torch
import torch.nn as nn
import numpy as np

from models.UNet_3Plus import UNet_3Plus_DeepSup_CGM #, UNet_3Plus_DeepSup
from loss.bceLoss import BCE_loss
from loss.iouLoss import IOU,IOU_loss
from loss.msssimLoss import MSSSIM

import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataloader.cocoform_loader import CustomDataLoader

import wandb


# In[2]:


# seed 고정
random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


# In[3]:


batch_size = 2
EPOCHS = 10

# In[4]:


common_json_path = "/opt/ml/segmentation/input/data/"
train_json_path = common_json_path + "train.json"
val_json_path = common_json_path + "val.json"
test_json_path = common_json_path + "test.json"


# In[5]:


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

train_transform = A.Compose([
                            ToTensorV2()
                            ])

val_transform = A.Compose([
                          ToTensorV2()
                          ])
'''
test_transform = A.Compose([
                           ToTensorV2()
                           ])
'''
# train dataset
train_dataset = CustomDataLoader(
    common_dir=common_json_path, json_path=train_json_path, 
    mode='train', transform=train_transform
)

# validation dataset
val_dataset = CustomDataLoader(
    common_dir=common_json_path, json_path=val_json_path, 
    mode='val', transform=val_transform
)
'''
# test dataset
test_dataset = CustomDataLoader(
    common_dir=common_json_path, json_path=test_json_path, 
    mode='test', transform=test_transform
)
'''
# DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, 
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)
'''
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    # num_workers=4,
    collate_fn=collate_fn
)
'''

# In[6]:


# UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM
model = UNet_3Plus_DeepSup_CGM(
    in_channels=3,
    n_classes=len(train_dataset.category_names),
    # feature_scale=4,
    is_deconv=True,
    is_batchnorm=True
)
model = model.cuda()


# In[7]:



# In[8]:

opt = torch.optim.Adam(
    params=model.parameters(), lr=1e-4, weight_decay=0.00001
)

bce_loss = nn.BCELoss()
fcl_loss = nn.BCELoss(reduction='none')
iou_loss = IOU(size_average=True)
ms_ssim = MSSSIM()


# In[9]:
wandb.init(
    entity='passion-ate',
    project='segmentation',
    name='UNet3+'
)
tr_cnt, val_cnt = 0, 0
for ep in range(EPOCHS):
    
    model.train()
    train_epoch_loss = 0
    for a_batch in train_loader:
        x, y, cls_label = list(map(torch.stack, a_batch))
        cls_branch, Dec = model(x.cuda())

        iteration_loss = 0
        iteration_loss += bce_loss(cls_branch, cls_label.cuda())
        # for D in Dec:
        #     iteration_loss += iou_loss(D, y.cuda())
        #     iteration_loss += ms_ssim(D, y.cuda())
        D = Dec[0]
        focal_loss = fcl_loss(D, y.cuda())*((1-D)**2)
        iteration_loss += focal_loss.mean()
        iteration_loss += iou_loss(D, y.cuda())
        iteration_loss += ms_ssim(D, y.cuda())
        train_epoch_loss +=iteration_loss
        
        opt.zero_grad()
        iteration_loss.backward()
        opt.step()
        tr_cnt += 1
        wandb.log(
            data={"UNet/train_iter_loss": iteration_loss},
            step=tr_cnt,
        )
    
    with torch.no_grad():
        model.eval()
        valid_epoch_loss = 0
        for a_batch in val_loader:
            x, y, cls_label = list(map(torch.stack, a_batch))
            cls_branch, Dec = model(x.cuda())

            iteration_loss = 0
            iteration_loss += bce_loss(cls_branch, cls_label.cuda())
            # for D in Dec:
            #     iteration_loss += iou_loss(D, y.cuda())
            #     iteration_loss += ms_ssim(D, y.cuda())
            D = Dec[0]
            focal_loss = fcl_loss(D, y.cuda())*((1-D)**2)
            iteration_loss += focal_loss.mean()
            iteration_loss += iou_loss(D, y.cuda())
            iteration_loss += ms_ssim(D, y.cuda())
            valid_epoch_loss +=iteration_loss
            val_cnt += 1
            wandb.log(
                data={"UNet/valid_iter_loss": iteration_loss},
                step=val_cnt,
            )
    
    wandb.log(
        data={
            "UNet/train_epoch_loss": train_epoch_loss/len(train_dataset),
            "UNet/val_epoch_loss": valid_epoch_loss/len(val_dataset)
        },
        step=ep
    )
# In[10]:




# In[ ]:




