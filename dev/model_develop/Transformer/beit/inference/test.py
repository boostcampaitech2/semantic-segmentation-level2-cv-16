import argparse
import os
import pandas as pd
import numpy as np
import json

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv import Config

from backbone import beit

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument("--csv", required=True, help="output file")
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    # parser.add_argument(
    #     '--eval',
    #     type=str,
    #     nargs='+',
    #     help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
    #     ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    # parser.add_argument(
    #     '--tmpdir',
    #     help='tmp directory used for collecting results from multiple '
    #     'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    # assert args.out or args.eval or args.format_only or args.show \
    #     or args.show_dir, \
    #     ('Please specify at least one operation (save/eval/format/show the '
    #      'results / save the results) with the argument "--out", "--eval"'
    #      ', "--format-only", "--show" or "--show-dir"')

    # if args.eval and args.format_only:
    #     raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # # set cudnn_benchmark
    # if cfg.get('cudnn_benchmark', False):
    #     torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    distributed = False
    # # init distributed env first, since logger depends on the dist info.
    # if args.launcher == 'none':
    #     distributed = False
    # else:
    #     distributed = True
    #     init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    # for data in data_loader:
    #     print(data)
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = False
    # if args.eval_options is not None:
        # efficient_test = args.eval_options.get('efficient_test', False)

    model = MMDataParallel(model.cuda(), device_ids=[0])
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, efficient_test)
    # print(outputs)
    # sample_submisson.csv 열기
    submission = pd.read_csv(
        "/opt/ml/segmentation/baseline_code/submission/sample_submission.csv",
        index_col=None,
    )
    json_dir = os.path.join("/opt/ml/segmentation/input/data/test.json")
    with open(json_dir, "r", encoding="utf8") as outfile:
        datas = json.load(outfile)

    input_size = 512
    output_size = 256
    bin_size = input_size // output_size

    # PredictionString 대입
    for image_id, predict in enumerate(outputs):
        image_id = datas["images"][image_id]
        file_name = image_id["file_name"]
        temp_mask = []
        predict = predict.reshape(1, 512, 512)
        mask = (
            predict.reshape((1, output_size, bin_size, output_size, bin_size)).max(4).max(2)
        )  # resize to 256*256
        temp_mask.append(mask)
        oms = np.array(temp_mask)
        oms = oms.reshape([oms.shape[0], output_size * output_size]).astype(int)

        string = oms.flatten()

        submission = submission.append(
            {
                "image_id": file_name,
                "PredictionString": " ".join(str(e) for e in string.tolist()),
            },
            ignore_index=True,
        )

    # submission.csv로 저장
    submission.to_csv(args.csv, index=False)
    
if __name__ == "__main__":
    main()