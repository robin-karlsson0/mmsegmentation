_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_feature_adaption.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_0k.py'
]