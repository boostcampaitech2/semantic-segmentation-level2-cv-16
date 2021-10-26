import os
import mmcv

from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

import pandas as pd
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--csv', required=True, help='output file')
parser.add_argument('--cfg', required=True, help='config file')
parser.add_argument('--ckpt', required=True, help='checkpoint file')
parser.add_argument('--tta', type=int, required=True, help='tta')
args = parser.parse_args()

cfg = Config.fromfile(args.cfg)
# root='../input/data/mmseg/image/test/'

epoch = 'best_mIoU_iter_28000'

# dataset config 수정
# cfg.data.test.img_dir = root
cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
if args.tta:
    cfg.data.test.pipeline[1]['img_ratios'] = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.] # TTA
    cfg.data.test.pipeline[1]['flip'] = True

cfg.data.test.test_mode = True

cfg.data.samples_per_gpu = 8

#cfg.work_dir = '/opt/ml/segmentation/semantic-segmentation-level2-cv-16/dev/model_develop/mmsegmentation/work_dirs/10-2'

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None

# checkpoint path
checkpoint_path = args.ckpt


dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])


output = single_gpu_test(model, data_loader)

# sample_submisson.csv 열기
submission = pd.read_csv('/opt/ml/segmentation/baseline_code/submission/sample_submission.csv', index_col=None)
json_dir = os.path.join("/opt/ml/segmentation/input/data/test.json")
with open(json_dir, "r", encoding="utf8") as outfile:
    datas = json.load(outfile)

input_size = 512
output_size = 256
bin_size = input_size // output_size
		
# PredictionString 대입
for image_id, predict in enumerate(output):
    image_id = datas["images"][image_id]
    file_name = image_id["file_name"]
    
    temp_mask = []
    predict = predict.reshape(1, 512, 512)
    mask = predict.reshape((1, output_size, bin_size, output_size, bin_size)).max(4).max(2) # resize to 256*256
    temp_mask.append(mask)
    oms = np.array(temp_mask)
    oms = oms.reshape([oms.shape[0], output_size*output_size]).astype(int)

    string = oms.flatten()

    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

# submission.csv로 저장
submission.to_csv(args.csv, index=False)