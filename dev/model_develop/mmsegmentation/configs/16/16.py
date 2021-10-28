# ocrnet_hr48_512x512_40k_voc12aug
_base_ = [
    "upernet_swin.py",
    "cocotrash.py",
    "default_runtime.py",
    "schedule_160k.py",
    # "pretrain_384x384_1K.py",
]

model = dict(
    pretrained="/opt/ml/segmentation/semantic-segmentation-level2-cv-16/dev/model_develop/mmsegmentation/configs/10/pretrain/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_converted.pth",
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
    ),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=11),
    auxiliary_head=dict(in_channels=512, num_classes=11),
)


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

lr_config = dict(
    _delete_=True,
    policy="CosineRestart",
    periods=[30, 50],
    restart_weights=[1.0, 0.5],
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.01,
    min_lr=6e-6,
    by_epoch=True,
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=8)
