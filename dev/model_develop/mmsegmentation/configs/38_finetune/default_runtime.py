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
                name="38_finetune-Upernet_SwinL_p4_w12_revisedv1",
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
load_from = None
resume_from = None
# resume_from = "/opt/ml/segmentation/semantic-segmentation-level2-cv-16/dev/model_develop/mmsegmentation/work_dirs/38_finetune/best_mIoU_iter_1500.pth"
workflow = [("train", 1)]
cudnn_benchmark = True
