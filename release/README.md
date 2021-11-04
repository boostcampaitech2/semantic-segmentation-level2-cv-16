
# 설치
- unlim의 beit git clone하기
`git clone https://github.com/microsoft/unilm.git`

- mmcv-full 설치
`pip install --ignore-installed mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html`

- mmsegmentation 설치
`pip install mmsegmentation==0.11.0`

# Train
`bash tools/dist_train.sh {config file} {number of GPU} --work-dir {work dir name} --seed {seed}`

# Inference
- 폴더에 있는 test.py로 변경
`bash tools/dist_test.sh {config file} {checkpoint} {number of GPU} --csv {csv file name}`