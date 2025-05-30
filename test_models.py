#!/usr/bin/env python

import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

from mmengine.registry import init_default_scope
init_default_scope('mmdet')

import mmdet
from mmdet.utils import register_all_modules

register_all_modules()

from mmengine.registry import TRANSFORMS, MODELS, TASK_UTILS, DATASETS, METRICS
from mmdet.datasets.transforms.loading import LoadAnnotations as DetLoadAnnotations, LoadImageFromFile
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.transforms import Resize, RandomFlip

TRANSFORMS.register_module(name='LoadAnnotations', module=DetLoadAnnotations, force=True)
TRANSFORMS.register_module(name='LoadImageFromFile', module=LoadImageFromFile, force=True)
TRANSFORMS.register_module(name='PackDetInputs', module=PackDetInputs, force=True)
TRANSFORMS.register_module(name='Resize', module=Resize, force=True)
TRANSFORMS.register_module(name='RandomFlip', module=RandomFlip, force=True)

from mmdet.models.detectors.mask_rcnn import MaskRCNN
from mmdet.models.detectors.sparse_rcnn import SparseRCNN
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.roi_heads.sparse_roi_head import SparseRoIHead
from mmdet.models.dense_heads.rpn_head import RPNHead
from mmdet.models.dense_heads.embedding_rpn_head import EmbeddingRPNHead
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead
from mmdet.models.roi_heads.bbox_heads.dii_head import DIIHead
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead
from mmdet.models.roi_heads.mask_heads.dynamic_mask_head import DynamicMaskHead
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.necks.fpn import FPN
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import SingleRoIExtractor

from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.task_modules.assigners.max_iou_assigner import MaxIoUAssigner
from mmdet.models.task_modules.samplers.random_sampler import RandomSampler
from mmdet.models.layers.transformer.utils import coordinate_to_encoding, inverse_sigmoid
from mmdet.models.layers.positional_encoding import SinePositionalEncoding

from mmdet.models.task_modules.prior_generators.anchor_generator import AnchorGenerator

from mmdet.models.task_modules.assigners.iou2d_calculator import BboxOverlaps2D

task_utils = [
    ('DeltaXYWHBBoxCoder', DeltaXYWHBBoxCoder),
    ('MaxIoUAssigner', MaxIoUAssigner),
    ('RandomSampler', RandomSampler),
    ('AnchorGenerator', AnchorGenerator),
    ('BboxOverlaps2D', BboxOverlaps2D),
]

for name, cls in task_utils:
    TASK_UTILS.register_module(name=name, module=cls, force=True)
    print(f"✅ 强制注册了任务工具{name}")

model_components = [
    ('MaskRCNN', MaskRCNN),
    ('SparseRCNN', SparseRCNN),
    ('StandardRoIHead', StandardRoIHead),
    ('SparseRoIHead', SparseRoIHead),
    ('RPNHead', RPNHead),
    ('EmbeddingRPNHead', EmbeddingRPNHead),
    ('Shared2FCBBoxHead', Shared2FCBBoxHead),
    ('DIIHead', DIIHead),
    ('FCNMaskHead', FCNMaskHead),
    ('DynamicMaskHead', DynamicMaskHead),
    ('ResNet', ResNet),
    ('FPN', FPN),
    ('DetDataPreprocessor', DetDataPreprocessor),
    ('SingleRoIExtractor', SingleRoIExtractor),
]

for name, cls in model_components:
    MODELS.register_module(name=name, module=cls, force=True)
    print(f"✅ 强制注册了{name}")

from mmdet.datasets import CocoDataset
from mmdet.evaluation.metrics import CocoMetric

from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import L1Loss

import torch.nn as nn

DATASETS.register_module(name='CocoDataset', module=CocoDataset, force=True)
METRICS.register_module(name='CocoMetric', module=CocoMetric, force=True)

MODELS.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss, force=True)
MODELS.register_module(name='L1Loss', module=L1Loss, force=True)
print("✅ 强制注册了损失函数 CrossEntropyLoss 和 L1Loss")

MODELS.register_module(name='Linear', module=nn.Linear, force=True)
MODELS.register_module(name='Conv2d', module=nn.Conv2d, force=True)
MODELS.register_module(name='BatchNorm2d', module=nn.BatchNorm2d, force=True)
MODELS.register_module(name='ReLU', module=nn.ReLU, force=True)
print("✅ 强制注册了基础PyTorch层")

from mmengine import Config
from mmengine.runner import Runner
import torch

def verify_model_registration():
    print("🔍 验证模型注册状态...")
    
    models_to_check = ['MaskRCNN', 'SparseRCNN', 'StandardRoIHead', 'SparseRoIHead', 'RPNHead', 'EmbeddingRPNHead']
    all_registered = True
    
    for model in models_to_check:
        if model in MODELS.module_dict:
            print(f"✅ {model} 已注册")
        else:
            print(f"❌ {model} 未注册")
            all_registered = False
    
    return all_registered

def test_model(config_file, checkpoint_file, test_type='bbox'):
    print(f"🧪 开始测试模型：{checkpoint_file}")
    print(f"📄 配置文件：{config_file}")
    print(f"🎯 测试类型：{test_type}")
    
    cfg = Config.fromfile(config_file)
    
    cfg.work_dir = os.path.dirname(checkpoint_file)
    
    runner = Runner.from_cfg(cfg)
    
    runner.load_checkpoint(checkpoint_file)
    
    runner.test()
    
    print(f"✅ 模型测试完成")

def evaluate_model_on_test_set(config_file, checkpoint_file):
    print(f"📊 在测试集上评估模型...")
    
    cfg = Config.fromfile(config_file)
    
    cfg.test_dataloader.dataset.ann_file = 'annotations/instances_test2017.json'
    cfg.test_dataloader.dataset.data_prefix.img = 'test2017/'
    
    cfg.test_evaluator.ann_file = cfg.data_root + 'annotations/instances_test2017.json'
    
    cfg.work_dir = os.path.dirname(checkpoint_file)
    
    runner = Runner.from_cfg(cfg)
    
    runner.load_checkpoint(checkpoint_file)
    
    runner.test()
    
    print(f"✅ 测试集评估完成")

def main():
    parser = argparse.ArgumentParser(description='测试训练好的模型')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, help='检查点文件路径')
    parser.add_argument('--test-set', action='store_true', help='在测试集上评估')
    
    args = parser.parse_args()
    
    if not verify_model_registration():
        print("❌ 模型注册验证失败")
        return
    
    print(f"🚀 开始模型测试...")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
    
    if args.config and args.checkpoint:
        if args.test_set:
            evaluate_model_on_test_set(args.config, args.checkpoint)
        else:
            test_model(args.config, args.checkpoint)
    else:
        model_configs = [
            ('mask_rcnn.py', 'work_dirs/mask_rcnn/epoch_24.pth'),
            ('sparse_rcnn.py', 'work_dirs/sparse_rcnn/epoch_12.pth'),
        ]
        
        for config_file, checkpoint_file in model_configs:
            if os.path.exists(config_file) and os.path.exists(checkpoint_file):
                print(f"\n{'='*50}")
                print(f"测试模型: {os.path.basename(checkpoint_file)}")
                print(f"{'='*50}")
                
                try:
                    if args.test_set:
                        evaluate_model_on_test_set(config_file, checkpoint_file)
                    else:
                        test_model(config_file, checkpoint_file)
                except Exception as e:
                    print(f"❌ 测试失败: {e}")
            else:
                print(f"⚠️ 跳过 {config_file} - 文件不存在")
    
    print("✅ 所有测试完成!")

if __name__ == '__main__':
    main() 