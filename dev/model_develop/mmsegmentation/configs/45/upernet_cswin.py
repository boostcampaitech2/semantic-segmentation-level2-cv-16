# model settings
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    pretrained=None,
    backbone=dict(
        type="CSWin",
        embed_dim=64,
        patch_size=4,
        depth=[1, 2, 21, 1],
        num_heads=[2, 4, 8, 16],
        split_size=[1, 2, 7, 7],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ),
    decode_head=dict(
        type="UPerHead",
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=0.75),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=0.25),
        ],
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=0.75 * 0.4),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=0.25 * 0.4),
        ],
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
