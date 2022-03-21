"""
Segmentation labels follow the Cityscapes convention with 19 classes.
        0:  road
        1:  sidewalk
        2:  building
        3:  wall
        4:  fence
        5:  pole
        6:  traffic light
        7:  traffic sign
        8:  vegetation
        9:  terrain
        10: sky
        11: person
        12: rider
        13: car
        14: truck
        15: bus
        16: train
        17: motorcycle
        18: bicycle

    Directory structure:
        bdd100k/
            seg/
                color_labels/
                    ...
                images/
                    test/
                        ac6d4f42-00000000.jpg
                        ...
                    train/
                        0a0a0b1a-7c39d841.jpg
                        ...
                    val/
                        7d2f7975-e0c1c5a7.jpg
                        ...
                labels/
                    train/
                        0a0a0b1a-7c39d841_train_id.png
                        ...
                    val/
                        7d2f7975-e0c1c5a7_train_id.png
                        ...
            ...

    Ref: https://doc.bdd100k.com/
"""

# dataset settings
dataset_type = 'BDD100KDataset'
# Symbolic link: bdd100k/seg --> data/bdd100k
data_root = 'data/bdd100k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (720, 720)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='labels/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='labels/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='labels/val',
        pipeline=test_pipeline))
