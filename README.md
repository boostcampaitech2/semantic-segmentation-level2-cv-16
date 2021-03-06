# Passion-ateπ₯
| [κ°μ¬ν](https://github.com/AshHyun) | [κΉλ―Όμ€](https://github.com/danny0628) | [λ°μν](https://github.com/hyun06000) | [μκ΄μ±](https://github.com/Gwang-chae) | [μ€νμ](https://github.com/Haeun-Oh) | [μ΄μΉμ°](https://github.com/DaleLeeCoding) |
| :-: | :-: | :-: | :-: | :-: | :-: |
| ![image](https://user-images.githubusercontent.com/65941859/137628452-e2f573fe-0143-46b1-925d-bc58b2317474.png) | ![image](https://user-images.githubusercontent.com/65941859/137628521-10453cac-ca96-4df8-8ca0-b5b0d00930c0.png) | ![image](https://user-images.githubusercontent.com/65941859/137628500-342394c3-3bbe-4905-984b-48fae5fc75d6.png) | ![image](https://user-images.githubusercontent.com/65941859/137628535-9afd4035-8014-475c-899e-77304950c190.png) | ![image](https://user-images.githubusercontent.com/65941859/137628474-e9c4ab46-0a51-4a66-9109-7462d3a7ead1.png) | ![image](https://user-images.githubusercontent.com/65941859/137628443-c032259e-7a7a-4c2d-891a-7db09b42d27b.png) |
|  | [Blog](https://danny0628.tistory.com/) | [Blog](https://davi06000.tistory.com/) |[Notion](https://kcseo25.notion.site/) |  | [Notion](https://leeseungwoo.notion.site/) |

<div align="center">

![python](http://img.shields.io/badge/Python-000000?style=flat-square&logo=Python)
![pytorch](http://img.shields.io/badge/PyTorch-000000?style=flat-square&logo=PyTorch)
![ubuntu](http://img.shields.io/badge/Ubuntu-000000?style=flat-square&logo=Ubuntu)
![git](http://img.shields.io/badge/Git-000000?style=flat-square&logo=Git)
![github](http://img.shields.io/badge/Github-000000?style=flat-square&logo=Github)
[![mmcv](http://img.shields.io/badge/MMCV-000000?style=flat-square)](https://github.com/open-mmlab/mmcv)
[![mmsegmentation](http://img.shields.io/badge/MMSegmentation-000000?style=flat-square)](https://github.com/open-mmlab/mmsegmentation)
[![smp](http://img.shields.io/badge/SMP-000000?style=flat-square)](https://github.com/open-mmlab/mmsegmentation)
</div align="center">


## Overview
νλ‘μ νΈ λͺ©ν  
- μ¬νμ© image semantic segmentation λͺ¨λΈ κ°λ°  

λ°μ΄ν°μ  
- COCO format μ΄λ―Έμ§ 3272μ₯ (512x512 resolution)
  - Class  : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
  - Tool: Pytorch, mmsegmentation, Weights and Biases, Github, Notion

## Result
- Public & Private LB 1λ±: 0. 773(19 teams)
![leaderboard](https://user-images.githubusercontent.com/65941859/140648164-1d0ae92f-7cb6-4c5b-8045-06650f5a7d25.png)

## Directory
```
semantic-segmentation-level2-cv-16/  
βββ dev/  
βββ release/  
    βββ data_processing/  
    |   βββ EDA_vis_tools/   
    |   βββ Passion-ateMix/   
    |   βββ Pseudo_labeling/   
    |   βββ ensemble/   
    |   βββ post_process_vis/   
    βββ segmentation_model/  
        βββ BEiT/  
        βββ HRNetV2_W64_OCR/  
        βββ SMP/  
        βββ SwinB-UperNet/  
        βββ UNet/  
```
<br></br>
## Model
### BEiT
[BEiT](./release/segmentation_model/BEiT)
### SwinTransformer
[SwinB-UperNet](./release/segmentation_model/SwinB-UperNet)
### HRNetV2
[HRNetV2_W64_OCR](./release/segmentation_model/HRNetV2_W64_OCR)
### UNet++
[SMP](./release/segmentation_model/SMP)


## Ensemble
[hard_voting.ipynb](./release/data_processing/ensemble)
