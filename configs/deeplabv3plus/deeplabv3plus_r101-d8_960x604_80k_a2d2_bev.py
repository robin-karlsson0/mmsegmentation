_base_ = './deeplabv3plus_r50-d8_960x604_80k_a2d2_bev.py'
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))

#model = dict(
#    decode_head=dict(
#        num_classes=2,
#        loss_decode=dict(
#            type='CrossEntropyLoss',
#            use_sigmoid=False,
#            loss_weight=1.0,
#            class_weight=[0.05, 1.0]),
#    ),
#    auxiliary_head=dict(
#        num_classes=2,
#        loss_decode=dict(
#            type='CrossEntropyLoss',
#            use_sigmoid=False,
#            loss_weight=0.4,
#            class_weight=[0.05, 1.0])),
#)

