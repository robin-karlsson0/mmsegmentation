_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/concat_bev.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
model = dict(
    #backbone=dict(
    #    frozen_stages=4,  # [stem: 0, resnet: 1,2,3,4]
    #),
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))

#model = dict(
#    decode_head=dict(
#        num_classes=2,
#        loss_decode=dict(
#            type='CrossEntropyLoss',
#            use_sigmoid=False,
#            loss_weight=1.0,
#            class_weight=[0.9159390842456511, 1.084060915754349]),
#    ),
#    auxiliary_head=dict(
#        num_classes=2,
#        loss_decode=dict(
#            type='CrossEntropyLoss',
#            use_sigmoid=False,
#            loss_weight=0.4,
#            class_weight=[0.9159390842456511, 1.084060915754349])),
#)

