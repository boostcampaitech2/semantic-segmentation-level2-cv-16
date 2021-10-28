import os
import warnings 
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

# GPU 사용 가능 여부에 따라 device 정보 저장
device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, common_dir, json_path, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(json_path)
        self.dataset_path = common_dir
        self.category_names = [
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
        ]

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
        _images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']))
        _images = cv2.cvtColor(_images, cv2.COLOR_BGR2RGB).astype(np.float32)
        # _images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            ground_truth = np.zeros(
                (
                    image_infos["height"],
                    image_infos["width"]
                )
            )
            masks = np.zeros(
                (
                    len(self.category_names),
                    image_infos["height"],
                    image_infos["width"]
                )
            )
            cls_onehots = torch.zeros(len(self.category_names))
            # General trash = 1, ... , Cigarette = 10
            # anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=False)
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            for i in range(len(anns)):
                className = self.get_classname(anns[i]['category_id'], cats)
                cate_idx = self.category_names.index(className)
                cls_onehots[cate_idx] = 1
                ground_truth[self.coco.annToMask(anns[i]) == 1] = cate_idx
            ground_truth = ground_truth.astype(np.int64)
            
            masks=[]
            for i in range(len(self.category_names)):
                buf = np.zeros(
                    (
                        image_infos["height"],
                        image_infos["width"]
                    )
                )
                buf[ground_truth==i] += 1
                masks.append(buf)
            masks = np.array(masks).astype(np.float32)
            
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                if masks.shape[-1] != len(self.category_names):
                    masks = np.transpose(masks,(1,2,0))
                
                transformed = self.transform(
                    image=_images,
                    mask0=masks,
                    mask1=ground_truth
                )
                images = transformed["image"]
                masks = transformed["mask0"]
                ground_truth = transformed["mask1"]
                
                H, W = ground_truth.shape
                Resize = A.Compose([
                    A.Resize(width=W//2, height=H//2),
                    ToTensorV2()
                ],additional_targets={
                    'image': 'image',
                    'mask0': 'mask',
                    'mask1': 'mask',
                })
                Resized = Resize(image=images, mask0=masks, mask1=ground_truth)
                qtr = Resized["image"]
                masks = Resized["mask0"]
                ground_truth = Resized["mask1"]
                images = ToTensorV2()(image=images)["image"]
                
                if masks.shape[0] != len(self.category_names):
                    masks = masks.permute(2,0,1)
                
            return images, masks, ground_truth.type(torch.int64), cls_onehots, qtr#, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=_images)
                images = transformed["image"]
            return images#, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())