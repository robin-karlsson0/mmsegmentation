_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_vissl.py',
    '../_base_/datasets/gta5.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
