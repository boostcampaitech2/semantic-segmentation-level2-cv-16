# BEiT
## 설치
- unlim의 beit git clone하기

`git clone https://github.com/microsoft/unilm.git`

- mmcv-full 설치

`pip install --ignore-installed mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html`

- mmsegmentation 설치

`pip install mmsegmentation==0.11.0`

- timm 설치

`pip install scipy timm==0.3.2`

## Pretrained model
[pretrained](https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k_ft22k.pth)

## Train
`bash tools/dist_train.sh {config file} {number of GPU} --work-dir {work dir name} --seed {seed}`

## Inference
- inference 폴더에 있는 test.py로 변경

`bash tools/dist_test.sh {config file} {checkpoint} {number of GPU} --csv {csv file name}`

## Config file
- 44: Default data에 대한 config file
- 56: Revised data에 대한 config file
- 61: pseudo labeling에 대한 config file
- 64: train_all에 pseudo labeling 적용한 config file
