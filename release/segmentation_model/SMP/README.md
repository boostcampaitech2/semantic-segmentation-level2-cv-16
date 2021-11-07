# Usage
## Setting
- requirements.txt 설치 
  - `pip install -r requirements.txt`

## Train
- `python train.py --seed 2021 --epochs 100 --batch_size 8`
  - `--seed` : random seed (default: 42)
  - `--epochs` : number of epochs to train (default: 25)
  - `--batch_size` : input batch size for training (default: 8)
  - `--lr` : learning rate (default: 5e-6)
  - `--name` : model save at {SM_MODEL_DIR}/{name}
  - `--log_every` : logging interval (default: 25)

- smp_model.py에 있는 myModel class를 통해 model 관리 -> 원하는 SMP model의 encoder, backbone, weight 등을 관리 가능
- best model에 사용
  - encoder : `timm-efficientnet-b4`
  - model : `UnetPlusPlus`
  - weight : `noisy-student`

## Dataloader
- `class CustomDataLoader` : train.py에서 사용된 DataLoader
- `class PseudoTrainset` : pseudo.py에서 사용된 DataLoader

## Inference & Inference_TTA
- 결과 csv file을 뽑아 내기위한 ipynb
