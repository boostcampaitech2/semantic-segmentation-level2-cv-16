_base_ = [
    "upernet_beit.py",
    "cocotrash.py",
    "default_runtime.py",
    "schedule_160k.py"
]
crop_size = (512, 512)
model = dict(
    pretrained = "/opt/ml/segmentation/unilm/beit/semantic_segmentation/configs/beit/pretrain/beit_large_patch16_224_pt22k_ft22k.pth",
    backbone=dict(
        type='BEiT',
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23],
    ),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024],
        num_classes=11,
        channels=1024,
    ),
    auxiliary_head=dict(
        in_channels=1024,
        num_classes=11
    ), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
# optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))

optimizer = dict(_delete_=True, type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)

find_unused_parameters = True

# runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )