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
                name="61-Upernet_Beit_Large_pseudo",
                entity="passion-ate",
            ),
        ),
    ],
)
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = "/opt/ml/unilm/beit/semantic_segmentation/work_dirs/61/latest.pth"
workflow = [("train", 1)]
cudnn_benchmark = True
