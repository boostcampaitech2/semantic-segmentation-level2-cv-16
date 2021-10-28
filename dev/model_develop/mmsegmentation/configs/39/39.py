_base_ = [
    "upernet_cswin.py",
    "cocotrash.py",
    "default_runtime.py",
    "schedule_160k.py",
]
model = dict(
    # pretrained="/opt/ml/upernet_cswin_base.pth",
    backbone=dict(
        type="CSWin",
        embed_dim=96,
        depth=[2, 4, 32, 2],
        num_heads=[4, 8, 16, 32],
        split_size=[1, 2, 7, 7],
        drop_path_rate=0.6,
        use_chk=False,
    ),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=11),
    auxiliary_head=dict(in_channels=384, num_classes=11),
)
# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
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
            "head": dict(lr_mult=3.0),
        }
    ),
)

lr_config = dict(
    _delete_=True,
    policy="CosineRestart",
    periods=[30000, 50000],
    restart_weights=[1.0, 0.3],
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr=1e-6,
    by_epoch=False,
)
seed = 16
