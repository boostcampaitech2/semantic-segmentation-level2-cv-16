## Usage

### 1. Creating virtual environment (optional)
All code was developed and tested with Python 2.8.2 (Anaconda) and PyTorch 1.7.1.

```bash
$ conda create -n segmentation python=2.8.2
$ conda activate segmentation
```


### 2. Installing dependencies
```bash
$ pip install -r segmentation_requirements.txt
```


### 3. Training

```bash
$ python segmentation_model/HRNetV2_W64_OCR/HRNetV2_W64_OCR/train.py
```

The training script has a number of command-line flags that you can use to configure the model architecture, hyperparameters, and input / output settings:
- `seed`: random seed. Default is `16`
- `epochs`: number of epochs to train. Default is `25`
- `batch_size`:input batch size for training. Default is `12`
- `lr`: learning rate. Default is `1e-5`
- `name`: name of the model in Wandb. 
- `log_every`: logging interval. Default is `25`
- `vis_every`: image logging interval. Default is `10`


