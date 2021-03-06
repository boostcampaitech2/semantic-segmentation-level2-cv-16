#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools import mask


# In[2]:


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"


def get_image_detail(dataset_path, image_idx, coco, category_names):
    idx = image_idx

    image_id = coco.getImgIds(imgIds=idx)
    image_infos = coco.loadImgs(image_id)[0]

    ann_ids = coco.getAnnIds(imgIds=image_infos['id'])
    anns = coco.loadAnns(ann_ids)

    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)

    masks = np.zeros((image_infos["height"], image_infos["width"]))

    anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
    for i in range(len(anns)):
        className = get_classname(anns[i]['category_id'], cats)
        pixel_value = category_names.index(className)
        masks[coco.annToMask(anns[i]) == 1] = pixel_value

    masks = masks.astype(np.int8)

    images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    
    return masks, images, image_infos


def find_background_space(bg_masks):
    background_mask = (bg_masks == 0)

    edge = 0
    graph = [[x for x in sub] for sub in background_mask]

    for x in range(1, len(graph)):
        for y in range(1, len(graph)):
            if graph[x][y] == 0:
                continue
            else:
                _min = min([graph[x][y-1], graph[x-1][y], graph[x-1][y-1]])
                graph[x][y] = _min + 1
                if edge < graph[x][y]:
                    edge = graph[x][y]

    flag = False

    for x in range(1, len(graph)):
        if flag: 
            break
        for y in range(1, len(graph)):
            if graph[x][y] == edge:
                empty_ymax, empty_xmax = x+1, y+1
                flag = True
                break
                
    return edge, empty_ymax, empty_xmax


def find_foreground_bbox(masks, images, mask_class):
    masks = (masks == mask_class).astype(np.int8)

    # masking??? ?????? ?????? src
    # src??? ??? background ??????

    fg = cv2.bitwise_and(images, images, mask=masks)

    ## ?????? ????????? bbox ?????? ??????
    ## class??? annotation??? ???????????? ??? ?????? ????????? for?????? ?????? bbox ??????

    ymin, ymax = 0, 0

    for y in range(len(fg)):
        if (masks[y].sum() != 0) and ymin == 0:
            ymin = y
        elif masks[y].sum() != 0:
            ymax = y        

    xmin, xmax = 0, 0

    for x in range(len(fg[0])):
        if (masks[:, x].sum() != 0) and xmin == 0:
            xmin = x
        elif masks[:, x].sum() != 0:
            xmax = x
    
    fg_bbox = (xmin, ymin, xmax, ymax)
    return fg, fg_bbox


def resize_foreground_backgroud(masks, fg, fg_bbox, empty_ymax, empty_xmax, edge):
    xmin, ymin, xmax, ymax = fg_bbox
    
    segmentation_roi = masks[ymin:ymax, xmin:xmax]
    if edge * edge > (ymax-ymin) * (xmax-xmin):
        resized_segmentation_roi = cv2.resize(segmentation_roi.astype(np.uint8), dsize=(edge, edge), interpolation=cv2.INTER_AREA)
    else:
        resized_segmentation_roi = cv2.resize(segmentation_roi.astype(np.uint8), dsize=(edge, edge), interpolation=cv2.INTER_CUBIC)
        
    segmentation_mask = np.zeros((512, 512))
    segmentation_mask[empty_ymax-edge:empty_ymax, empty_xmax-edge:empty_xmax] = resized_segmentation_roi

    # mask ?????? ??????
    extract_mask = fg[ymin:ymax, xmin:xmax]

    # ????????? size??? ?????? resize
    if edge * edge > (ymax-ymin) * (xmax-xmin):
        resized_fg = cv2.resize(extract_mask, dsize=(edge, edge), interpolation=cv2.INTER_AREA)
    else:
        resized_fg = cv2.resize(extract_mask, dsize=(edge, edge), interpolation=cv2.INTER_CUBIC)
        
    return resized_fg, segmentation_mask


def merge_bg_fg(images, resized_fg, empty_ymax, empty_xmax, edge):
    # ?????? ??????????????? ????????????(roi) ??????
    # roi??? ?????? ??????????????? ?????? ??? ??????
    roi = images[empty_ymax-edge:empty_ymax, empty_xmax-edge:empty_xmax]

    # ????????? ????????? ?????? bg_mask, fg_mask ??????
    resized_fg_gray = cv2.cvtColor(resized_fg, cv2.COLOR_BGR2GRAY)

    bg_mask = (resized_fg_gray == 0).astype(np.int8)
    fg_mask = (resized_fg_gray != 0).astype(np.int8)

    # roi?????? ?????? ?????? ????????? ????????? ????????? ?????? ??????
    # resized_src?????? ?????? ?????? ????????? ??????
    bg = cv2.bitwise_and(roi, roi, mask=bg_mask)
    fg = cv2.bitwise_and(resized_fg, resized_fg, mask=fg_mask)

    # ????????? ??????
    merged_image = cv2.bitwise_or(bg, fg)

    # ?????? ???????????? ????????? ????????? ?????????
    images[empty_ymax-edge:empty_ymax, empty_xmax-edge:empty_xmax] = merged_image
    
    return images


def save_merged_image(save_dir, file_name_dir, merged_image, bg_image_infos):
    if not os.path.isdir(save_dir):                                                           
        os.mkdir(save_dir)

    merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_dir + bg_image_infos['file_name'].split('/')[-1], merged_image)
    
    
def make_coco_annotation(file_name_dir, bg_image_infos, segmentation_mask, mask_class, json_data):
    # ??????????????? image_id
    revised_id = bg_image_infos['id']

    # json?????? ???????????? image ??????
    json_data['images'][revised_id]['file_name'] = file_name_dir + bg_image_infos['file_name'].split('/')[-1]


    fortran_ground_truth_binary_mask = np.asfortranarray(segmentation_mask).astype(np.uint8)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours, hierarchy = cv2.findContours(segmentation_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": bg_image_infos['id'],
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": mask_class,
            "id": json_data['annotations'][-1]['id'] + 1
        }


    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()

        if len(segmentation) <= 4:
            continue
        annotation['segmentation'].append(segmentation)
        
    return json_data, annotation

