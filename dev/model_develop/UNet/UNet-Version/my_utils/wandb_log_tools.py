import wandb

def log_mask_img_wandb(sample_list, model, mode, class_to_labels):
    
    mask_img = []
    for sample in sample_list:
        img = sample[0]
        _img = img.permute(1,2,0).detach().numpy()
        gt = sample[2].detach().numpy()
        prediction = model(img.unsqueeze(0).cuda())
        if not isinstance(prediction, (list, tuple)):
            prediction = [prediction,]
            cls_branch = None
        else:
            cls_branch, prediction = prediction
        
        mask_dict = {
            "ground_truth": {    
                "mask_data": gt,    
                "class_labels": class_to_labels  
            }
        }
        
        
        for idx, decoded_img in enumerate(prediction):
            mask_name = f"dec-{idx}-prediction"
            decoded_img = decoded_img.max(dim=1)[1][0].cpu().detach().numpy().astype("int")
            decoded_masks={
                mask_name: {
                    "caption":mask_name,
                    "mask_data": decoded_img,    
                    "class_labels": class_to_labels  
                },
            }
            mask_dict.update(decoded_masks)

        mask_img += [wandb.Image(
            img,
            caption=f"{mode} sample",
            masks=mask_dict,
        )]
    wandb.log({f"{mode}-images-with-masks" : mask_img})

def sample_mask_img_wandb(sample_list, class_to_labels, mode):
    
    for sample in sample_list:
        x, y, gt, _ = sample

        img = x.permute(1,2,0).detach().numpy()
        y = y.detach().numpy()
        gt = gt.detach().numpy()

        mask_dict = {
            "ground_truth": {    
                "mask_data": gt,    
                "class_labels": class_to_labels  
            }
        }
        for i, cls in enumerate(class_to_labels):
            mask_dict.update(
                {
                    f"{cls}": {    
                        "mask_data": y[i],
                        "class_labels": {0:"0",1:"1"}
                    }
                }
            )
        
        mask_img = wandb.Image(
                img,
                caption=f"{mode} sample",
                masks=mask_dict,
            )
        wandb.log({f"{mode}-sample" : mask_img})