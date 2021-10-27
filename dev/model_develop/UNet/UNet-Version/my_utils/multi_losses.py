
import wandb
import torch.nn as nn


from loss.bceLoss import BCE_loss
from loss.iouLoss import IOU,IOU_loss
from loss.msssimLoss import MSSSIM
from my_utils.fcl_loss import FocalLoss

# import pytorch_msssim

class MultiLosses:
    def __init__(self):
        self.bce_loss = nn.BCELoss()
        self.fcl_loss = FocalLoss(gamma=140)#nn.CrossEntropyLoss(reduction='none')
        self.iou_loss = IOU(size_average=True)
        self.ms_ssim = MSSSIM()

    def init_loss_values(self):
        self._iou_loss_value = 0.0
        self._ms_ssim_loss_value = 0.0
        self._fcl_loss_value = 0.0
        self._bce_loss_value = 0.0

    
    def __call__(
        self, prediction, 
        y, gt=None, 
        cls_label=None,
        deep_super = True,
    ):
        '''
        if not isinstance(prediction,(list,tuple)):
            prediction = [prediction,]
            cls_branch = None
        else:
            cls_branch, prediction = prediction
        
        if not deep_super:
            prediction = [prediction[0]]
        '''
        cls_branch, prediction = prediction
        self.init_loss_values()
        for decoded_img in prediction:
            self._iou_loss_value += 0.00005*self.iou_loss(decoded_img, y.cuda())
            self._ms_ssim_loss_value += -1*0.00005*self.ms_ssim(decoded_img, y.cuda())
            if gt is not None:
                self._fcl_loss_value = 0.9999*self.fcl_loss(decoded_img, gt.cuda())
            else:
                self._fcl_loss_value += 0
            
            if not deep_super:
                break
            
        if cls_label is not None:
            self._bce_loss_value += self.bce_loss(cls_branch, cls_label.cuda())
        else:
            self._bce_loss_value += 0

        self.iteration_loss = (self._iou_loss_value + self._ms_ssim_loss_value + 
                            self._fcl_loss_value + self._bce_loss_value)
        return self.iteration_loss
    
    def wandb_log_step(self, step=None):
        if step is None:
            raise ValueError("A step for wandb log must be setted")
        wandb.log(
            data={
                "UNet/train_iter_loss": self.iteration_loss,
                "UNet/train_iou_loss": self._iou_loss_value,
                "UNet/train_ms_ssim_loss": self._ms_ssim_loss_value,
                "UNet/train_fcl_loss": self._fcl_loss_value,
                "UNet/train_cls_loss": self._bce_loss_value,
                "tr_step":step
            }
        )
    
    def get_loss_components(self):
        return (self._iou_loss_value, self._ms_ssim_loss_value, 
                self._fcl_loss_value, self._bce_loss_value)
