# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="segmentation",
                name="39-Upernet_CSWinB_p4_w12_revisedv1",
                entity="passion-ate",
            ),
            by_epoch=False,
            with_step=False,
        ),
    ],
)
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
# load_from = "/opt/ml/upernet_cswin_base.pth"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True