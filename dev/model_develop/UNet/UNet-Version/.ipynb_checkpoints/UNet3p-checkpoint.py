#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import random

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from my_utils.multi_losses import MultiLosses
from my_utils.wandb_log_tools import log_mask_img_wandb
from models.UNet_3Plus import (UNet_3Plus_DeepSup_CGM,
                                 UNet_3Plus_DeepSup, UNet_3Plus)

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
EPOCHS = 30

# In[4]:


common_json_path = "/opt/ml/segmentation/input/data/"
train_json_path = common_json_path + "train_v1.json"
val_json_path = common_json_path + "valid_v1.json"
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
train_sample = train_dataset[0]
# validation dataset
val_dataset = CustomDataLoader(
    common_dir=common_json_path, json_path=val_json_path, 
    mode='val', transform=val_transform
)
val_sample = val_dataset[0]
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


# UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM
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

class_dict_path = "/opt/ml/segmentation/baseline_code/class_dict.csv"
class_colormap = pd.read_csv(class_dict_path)
class_to_labels ={idx:cls_name for idx, cls_name in enumerate(class_colormap["name"])}
labels_to_class ={cls_name:idx for idx, cls_name in enumerate(class_colormap["name"])}

multi_loss = MultiLosses()

opt = torch.optim.Adam(
    params=model.parameters(), lr=3e-4, #weight_decay=0.001
)

# In[9]:
wandb.init(
    entity='sang-hyun',
    project='seg-unet3p',
    name='UNet3+'
)
iter_freq = 100
tr_cnt, val_cnt = 0, 0
for ep in range(EPOCHS):
    
    model.train()
    train_epoch_loss = 0
    for a_batch in train_loader:
        x, y, gt, cls_label = list(map(torch.stack, a_batch))
        prediction = model(x.cuda())
        
        iteration_loss = multi_loss(
            prediction=prediction,
            y=y, 
            gt=gt, 
            # cls_label=cls_label,
        )
        train_epoch_loss += iteration_loss
        
        opt.zero_grad()
        iteration_loss.backward()
        opt.step()
        if not (tr_cnt % iter_freq):
            multi_loss.wandb_log_step(step=tr_cnt)
            log_mask_img_wandb(
                sample=train_sample, 
                model=model, 
                mode="train", 
                class_to_labels=class_to_labels
            )
            
        tr_cnt += 1
        
    with torch.no_grad():
        model.eval()
        valid_epoch_loss = 0
        for a_batch in val_loader:
            x, y, gt, cls_label = list(map(torch.stack, a_batch))
            prediction = model(x.cuda())

            iteration_loss = multi_loss(
                prediction=prediction,
                y=y, 
                gt=gt, 
                # cls_label=cls_label,
            )
            valid_epoch_loss +=iteration_loss
        
        log_mask_img_wandb(
            sample=val_sample, 
            model=model, 
            mode="valid", 
            class_to_labels=class_to_labels
        )
            
    wandb.log(
        data={
            "UNet/train_epoch_loss": train_epoch_loss/len(train_dataset),
            "UNet/val_epoch_loss": valid_epoch_loss/len(val_dataset),
            "epoch":ep
        },
    )
    torch.save(model,f"./saved_model/epoch_{ep}.pt")
# In[10]:




# In[ ]:




