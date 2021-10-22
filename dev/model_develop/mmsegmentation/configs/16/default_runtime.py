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
                name="16-Upernet_SwinB_p4_w12_cosine_byepoch",
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
resume_from = "/opt/ml/segmentation/semantic-segmentation-level2-cv-16/dev/model_develop/mmsegmentation/work_dirs/16/latest.pth"
workflow = [("train", 1)]
cudnn_benchmark = True
