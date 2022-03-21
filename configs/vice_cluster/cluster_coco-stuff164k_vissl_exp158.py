_base_ = [
    '../_base_/models/cluster_vissl.py',
    '../_base_/datasets/coco-stuff164k_coarse.py',  # _nocrop_lowres.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_12k.py'
]
model = dict(
    decode_head=dict(
        in_channels=64,
        channels=128,
        num_convs=1,
        kernel_size=128,  # FEATURE DIM
        num_classes=27,
        act_cfg=None),
    vissl_params=dict(
        config_path=('/home/robin/projects/vissl/experiments/low_res/'
                     'sc_exp158/vice_8node_fpn_resnet18_coco_exp158.yaml'),
        checkpoint_path=('/home/robin/projects/vissl/experiments/low_res/'
                         'sc_exp158/model_final_checkpoint_phase3.torch'),
    ))

optimizer = dict(lr=0.01)  # default0.01
