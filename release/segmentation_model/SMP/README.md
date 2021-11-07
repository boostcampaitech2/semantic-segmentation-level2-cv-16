## Setting
- requirements.txt 설치 
  - `pip install -r requirements.txt`

## Train
- 실행 방법
  - `python train.py --seed 2021 --epochs 100 --batch_size 8`

- smp_model.py에 있는 myModel class를 통해 model 관리 -> 원하는 SMP model의 encoder, backbone, weight 등을 관리 가능

## Inference & Inference_TTA
- 결과 csv file을 뽑아 내기위한 ipynb
