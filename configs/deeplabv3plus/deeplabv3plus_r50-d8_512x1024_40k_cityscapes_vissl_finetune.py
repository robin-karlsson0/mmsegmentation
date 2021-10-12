_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_vissl_finetune.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# Layer group specific LR multipliers
optimizer = dict(
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))
