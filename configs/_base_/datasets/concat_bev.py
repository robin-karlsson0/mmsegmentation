dataset_a2d2_train = dict(
    type='A2D2DatasetBEV',
    data_root = 'data/a2d2/',
    img_dir = 'images/train',
    ann_dir = 'annotations/train',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1920, 604), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=(960, 604), cat_max_ratio=1.0),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        dict(type='Pad', size=(960, 604), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]
),
dataset_a2d2_val = dict(
    type='A2D2DatasetBEV',
    data_root = 'data/a2d2/',
    img_dir = 'images/val',
    ann_dir = 'annotations/val',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1920, 604),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
        ]
),
dataset_a2d2_test = dict(
    type='A2D2DatasetBEV',
    data_root = 'data/a2d2/',
    img_dir = 'images/test',
    ann_dir = 'annotations/test',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1920, 604),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
        ]
),


dataset_apolloscape_train = dict(
    type='ApolloscapeLanesegDatasetBEV',
    data_root = 'data/apolloscape_laneseg/',
    img_dir = 'images/train',
    ann_dir = 'annotations/train',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1692, 505), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=(1150, 505), cat_max_ratio=1.0),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        dict(type='Pad', size=(1150, 505), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ]
),

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train = [
        dataset_a2d2_train,
        dataset_apolloscape_train
    ],
    val = dataset_a2d2_val,
    test = dataset_a2d2_test
    )
