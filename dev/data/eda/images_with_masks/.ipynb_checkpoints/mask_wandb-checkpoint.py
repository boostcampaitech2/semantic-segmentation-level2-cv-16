#!/usr/bin/env python
# coding: utf-8

# # images with masks on W&B

# In[1]:


import os
import time
import json


import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import wandb
from pycocotools.coco import COCO
from easydict import EasyDict as edict


# In[2]:


common_json_path = "/opt/ml/segmentation/input/data/"
train_all_json_path = common_json_path + "train_all.json"
train_json_path = common_json_path + "train.json"
val_json_path = common_json_path + "val.json"


# In[3]:


# Read annotations
with open(val_json_path, 'r') as f:
    dataset = json.loads(f.read())
dataset = edict(dataset)

# In[4]:


category_names = []
for category in dataset.categories:
    category_names.append(category.name)
category_names


# In[5]:


class ImagesWithMasks:
    def __init__(self, data_dir, labels_to_class):
        self.coco = COCO(data_dir)
        self.labels_to_class = labels_to_class
    
    def get_classname(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = Image.open(os.path.join(common_json_path, image_infos['file_name']))
        

        ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
        anns = self.coco.loadAnns(ann_ids)

        # Load the categories in a variable
        cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(cat_ids)

        # masks : size가 (height x width)인 2D
        # 각각의 pixel 값에는 "category id" 할당
        # Background = 0
        masks = np.zeros((image_infos["height"], image_infos["width"]))
        # General trash = 1, ... , Cigarette = 10
        # anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=False)
        anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
        for i in range(len(anns)):
            className = self.get_classname(anns[i]['category_id'], cats)
            pixel_value = self.labels_to_class[className]
            masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
        masks = masks.astype(np.int8)

        return images, masks, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


# In[6]:


class_dict_path = "/opt/ml/segmentation/baseline_code/class_dict.csv"
class_colormap = pd.read_csv(class_dict_path)
class_colormap


# In[7]:


class_to_labels ={idx:cls_name for idx, cls_name in enumerate(class_colormap["name"])}
labels_to_class ={cls_name:idx for idx, cls_name in enumerate(class_colormap["name"])}


# In[8]:

train_all_loader = ImagesWithMasks(train_all_json_path,labels_to_class)


# In[16]:


Run = False
log_count, data_loader = 0, train_all_loader
for i in range(len(data_loader)):
    if log_count%100 == 0:
        if bool(Run):
            Run.finish()
        run_name = f"area-train-all-{log_count}-{log_count+99}"
        Run = wandb.init(
            reinit=True, 
            project="images-with-masks",
            entity="passion-ate",
            name=run_name,
        )
    
    img, mask, image_infos = data_loader[i]

    mask_img = wandb.Image(
        img,
        caption=image_infos["file_name"],
        masks={
            "ground_truth": {    
                "mask_data": mask,    
                "class_labels": class_to_labels  
            },
        }
    )
    Run.log({"images-with-masks" : mask_img})
    log_count += 1
    time.sleep(0.01)
        


# In[11]:

print("done")



# In[ ]:




