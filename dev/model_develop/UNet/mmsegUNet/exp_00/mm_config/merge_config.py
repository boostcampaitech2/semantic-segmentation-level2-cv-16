# author : Sang-hyun Park

from mmcv import Config

# default_config_path is setted as ralated path from exp_train.py
config_file_path = "pspnet_unet_s5-d16_128x128_40k_stare.py"
cfg = Config.fromfile(config_file_path)

cfg.dump("default_config.py")