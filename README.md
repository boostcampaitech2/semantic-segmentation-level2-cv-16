# semantic-segmentation-level2-cv-16
semantic-segmentation-level2-cv-16 created by GitHub Classroom


------------------

# Pseudo Labeling 실행방법
1. make_npy_from_csv.ipynb 를 실행하여 name.npy 와 mask.npy를 생성한다
2. 기존 dataloader에서 pseudo class를 추가하여 만든 dataloader_include_pseudo.py로 바꿔준다
3. pseudo.py를 실행한다. (train.py와 같은 arg 명령어를 치면 된다)
