# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True),
        # dict(type='TensorboardLoggerHook')
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="segmentation",
                name="21-Upernet_Beit",
                entity="passion-ate",
            ),
            by_epoch=True,
        ),
    ],
)
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True