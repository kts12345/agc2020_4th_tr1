_base_ = './config.py'
model = dict(
    pretrained = None
)
data = dict(
    workers_per_gpu = 1,
    test = dict(
        samples_per_gpu = 12,
        ann_file   = '/aichallenge/temp_dir/4th_anno.json',
        img_prefix = '/aichallenge/temp_dir/4th_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1920, 1080),
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]
    )
)

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.80),
    max_per_img=100)