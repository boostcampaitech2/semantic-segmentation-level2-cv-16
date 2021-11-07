# author : Sang-hyun Park

from mmcv import Config

# default_config_path is setted as ralated path from exp_train.py
config_file_path = "setr_naive_512x512_160k_b16_ade20k.py"
cfg = Config.fromfile(config_file_path)

cfg.dump("default_config.py")
