# optimizer
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy="poly", power=0.9, min_lr=1e-5, by_epoch=True)
# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=50)
checkpoint_config = dict(max_keep_ckpts=2, by_epoch=True, interval=1)
evaluation = dict(interval=1, metric="mIoU", pre_eval=True)
