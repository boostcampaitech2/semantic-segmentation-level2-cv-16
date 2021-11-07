# Train
`python train.py --seed 2021 --epochs 100 --batch_size 8`

myModel class를 통해 model 관리 -> 원하는 SMP model의 encoder, backbone, weight 등을 관리 가능

# Pseudo
- make_npy_from_csv.ipynb를 실행하여 img_name과 mask를 numpy array로 저장
- Pseudo folder에 pseudo.py를 실행

`python pseudo.py --seed 2021 --epochs 100 --batch_size 8`

