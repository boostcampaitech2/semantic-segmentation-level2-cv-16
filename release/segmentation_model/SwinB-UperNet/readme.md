# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/microsoft/Swin-Transformer">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/swin.py#L524">Code Snippet</a>

<details>
<summary align="right"><a href="https://arxiv.org/abs/2103.14030">Swin Transformer (arXiv'2021)</a></summary>

```latex
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

</details>

## Environment

`requirements.txt`를 이용하여 필요한 라이브러리를 설치합니다.
```shell
pip install -r requirements.txt
```

## Usage

pretrained 모델을 사용하시려면 mmsegmentation의 [swin2mmseg.py](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/model_converters/swin2mmseg.py)를 사용하여 [the official repo](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)에 있는 pretrained 모델을 mmsegmentation style key로 변환하셔야 합니다.
아래의 script는 `PRETRAIN_PATH` 에 있는 모델을 `STORE_PATH`에 저장합니다..
```shell
python tools/model_converters/swin2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```
This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.
현재 repository에 있는 config에서 사용하는 pretrained model은 아래에서 받으실 수 있습니다.
|config|model|
|------|-----|
|10_SwinB-UperNet|[pretrained](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth)|
|12_SwinB-UperNet|[pretrained](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth)|


