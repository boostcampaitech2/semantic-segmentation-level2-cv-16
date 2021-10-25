import wandb

def log_mask_img_wandb(sample, model, mode, class_to_labels):
    img = sample[0]
    _img = sample[0].permute(1,2,0).detach().numpy()
    gt = sample[2].detach().numpy()
    prediction = model(img.unsqueeze(0).cuda())
    prediction = prediction.max(dim=1)[1][0].cpu().detach().numpy()
    mask_img = wandb.Image(
        _img,
        caption=f"{mode} sample",
        masks={
            "ground_truth": {    
                "mask_data": gt,    
                "class_labels": class_to_labels  
            },
            "prediction": {    
                "mask_data": prediction,    
                "class_labels": class_to_labels  
            },
        }
    )
    wandb.log({f"{mode}-images-with-masks" : mask_img})
