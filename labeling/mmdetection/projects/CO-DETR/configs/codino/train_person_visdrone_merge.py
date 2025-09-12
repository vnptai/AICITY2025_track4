import sys
sys.path.append('/home/hungdv/mmdetection')

_base_ = ['co_dino_5scale_r50_8xb2_1x_coco.py']

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6))))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1408), (512, 1408), (544, 1408), (576, 1408),
                            (608, 1408), (640, 1408), (672, 1408), (704, 1408),
                            (736, 1408), (768, 1408), (800, 1408), (832, 1408),
                            (864, 1408), (896, 1408), (928, 1408), (960, 1408),
                            (992, 1408), (1024, 1408), (1056, 1408),
                            (1088, 1408), (1120, 1408), (1152, 1408),
                            (1184, 1408), (1216, 1408), (1248, 1408),
                            (1280, 1408), (1312, 1408), (1344, 1408),
                            (1376, 1408), (1408, 1408)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1408), (512, 1408), (544, 1408), (576, 1408),
                            (608, 1408), (640, 1408), (672, 1408), (704, 1408),
                            (736, 1408), (768, 1408), (800, 1408), (832, 1408),
                            (864, 1408), (896, 1408), (928, 1408), (960, 1408),
                            (992, 1408), (1024, 1408), (1056, 1408),
                            (1088, 1408), (1120, 1408), (1152, 1408),
                            (1184, 1408), (1216, 1408), (1248, 1408),
                            (1280, 1408), (1312, 1408), (1344, 1408),
                            (1376, 1408), (1408, 1408)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

data_root = '/home/hungdv/'
metainfo = {
    'classes': ('Pedestrian'),
}
dataset_type = 'CocoDataset'

train_dataloader = dict(
    batch_size=1, num_workers=1,
    dataset=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + 'visdrone/train_person_visdrone_fisheye_merge.json',
        data_prefix=dict(img=data_root + 'train_visdrone_fisheye_merge/images/')
    ))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + 'ms_coco-format_labels/test_person.json',
        data_prefix=dict(img=data_root + 'test/images/')
        ))
test_dataloader = val_dataloader

val_evaluator = dict(  # Validation evaluator config
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file=data_root + 'ms_coco-format_labels/test_person.json',  # Annotation file path
    metric=['bbox'],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False)
test_evaluator = val_evaluator  # Testing evaluator config

optim_wrapper = dict(optimizer=dict(lr=1e-4))

max_epochs = 16
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8],
        gamma=0.1)
]
