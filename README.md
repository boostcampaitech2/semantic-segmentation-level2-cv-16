# Passion-ate🔥
| [강재현](https://github.com/AshHyun) | [김민준](https://github.com/danny0628) | [박상현](https://github.com/hyun06000) | [서광채](https://github.com/Gwang-chae) | [오하은](https://github.com/Haeun-Oh) | [이승우](https://github.com/DaleLeeCoding) |
| :-: | :-: | :-: | :-: | :-: | :-: |
| ![image](https://user-images.githubusercontent.com/65941859/137628452-e2f573fe-0143-46b1-925d-bc58b2317474.png) | ![image](https://user-images.githubusercontent.com/65941859/137628521-10453cac-ca96-4df8-8ca0-b5b0d00930c0.png) | ![image](https://user-images.githubusercontent.com/65941859/137628500-342394c3-3bbe-4905-984b-48fae5fc75d6.png) | ![image](https://user-images.githubusercontent.com/65941859/137628535-9afd4035-8014-475c-899e-77304950c190.png) | ![image](https://user-images.githubusercontent.com/65941859/137628474-e9c4ab46-0a51-4a66-9109-7462d3a7ead1.png) | ![image](https://user-images.githubusercontent.com/65941859/137628443-c032259e-7a7a-4c2d-891a-7db09b42d27b.png) |
|  | [Blog](https://danny0628.tistory.com/) | [Blog](https://davi06000.tistory.com/) |[Notion](https://kcseo25.notion.site/) |  | [Notion](https://leeseungwoo.notion.site/) |

## Overview
프로젝트 목표: 재활용 객체 검출 모델 개발
데이터셋
COCO format 이미지 3272장 (512x512 resolution)
Class  : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
   - Tool: Pytorch, mmsegmentation, Weights and Biases, Github, Notion

## Result
- Public & Private LB 1등: 0. 773(19 teams)
![leaderboard](https://user-images.githubusercontent.com/65941859/140648164-1d0ae92f-7cb6-4c5b-8045-06650f5a7d25.png)

## Directory
semantic-segmentation-level2-cv-16/
├── dev/
├── release/
|   ├── data_processing/
|   |   ├── EDA_vis_tools/ 
|   |   ├── Passion-ateMix/ 
|   |   ├── Pseudo_labeling/ 
|   |   ├── ensemble/ 
|   |   ├── post_process_vis/ 
|   ├── segmentation_model/
|   |   |   ├── BEiT
|   |   |   ├── HRNetV2_W64_OCR
|   |   |   ├── SMP
|   |   |   ├── SwinB-UperNet
|   |   |   ├── UNet

## Model
### BEiT
[BEiT](./release/segmentation_model/BEiT)
### SwinTransformer
[SwinB-UperNet](./release/segmentation_model/SwinB-UperNet)
###HRNetV2
[HRNetV2_W64_OCR](./release/segmentation_model/HRNetV2_W64_OCR)
###UNet++
[SMP](./release/segmentation_model/SMP)

## Ensemble
[hard_voting.ipynb](./release/data_processing/ensemble)
