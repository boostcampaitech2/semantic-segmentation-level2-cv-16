import segmentation_models_pytorch as smp
import torch 
from importlib import import_module

def myModel(seg_model, encoder_name='timm-efficientnet-b4' ):
    smp_model =getattr(smp,seg_model)
    model =  smp_model(
                 encoder_name=encoder_name,
                 encoder_weights='noisy-student',
                 in_channels=3,
                 classes=11)
    return model