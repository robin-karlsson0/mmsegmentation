_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/kitti360.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
