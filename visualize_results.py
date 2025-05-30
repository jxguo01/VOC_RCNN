#!/usr/bin/env python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import random
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from mmengine.registry import init_default_scope
init_default_scope('mmdet')

import mmdet
from mmdet.utils import register_all_modules
register_all_modules()

from mmdet.apis import init_detector, inference_detector
from mmengine import Config
import torch

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

VOC_CLASSES_CN = [
    '飞机', '自行车', '鸟', '船', '瓶子', '巴士', '汽车',
    '猫', '椅子', '牛', '餐桌', '狗', '马', '摩托车',
    '人', '盆栽', '羊', '沙发', '火车', '电视'
]

def load_models():
    print("🚀 加载训练好的模型...")
    
    models = {}
    
    mask_rcnn_config = "mask_rcnn_low_memory.py"
    mask_rcnn_checkpoint = "work_dirs/mask_rcnn_restart_clean/epoch_12.pth"
    
    if os.path.exists(mask_rcnn_config) and os.path.exists(mask_rcnn_checkpoint):
        try:
            models['mask_rcnn'] = init_detector(mask_rcnn_config, mask_rcnn_checkpoint, device='cuda:0')
            print("✅ Mask R-CNN 加载成功")
        except Exception as e:
            print(f"❌ Mask R-CNN 加载失败: {e}")
    
    sparse_rcnn_config = "sparse_rcnn_low_memory.py"
    sparse_rcnn_checkpoint = "work_dirs/sparse_rcnn_low_memory/epoch_12.pth"
    
    if os.path.exists(sparse_rcnn_config) and os.path.exists(sparse_rcnn_checkpoint):
        try:
            models['sparse_rcnn'] = init_detector(sparse_rcnn_config, sparse_rcnn_checkpoint, device='cuda:0')
            print("✅ Sparse R-CNN 加载成功")
        except Exception as e:
            print(f"❌ Sparse R-CNN 加载失败: {e}")
    
    return models

def get_test_images(num_images=4):
    test_dir = "data/coco/test2017"
    if not os.path.exists(test_dir):
        print(f"❌ 测试图像目录不存在: {test_dir}")
        return []
    
    images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    if len(images) < num_images:
        print(f"⚠️ 测试图像数量不足，仅找到 {len(images)} 张")
        return [os.path.join(test_dir, img) for img in images]
    
    selected = random.sample(images, num_images)
    return [os.path.join(test_dir, img) for img in selected]

def draw_bboxes_and_masks(image, result, title="检测结果", score_threshold=0.3):
    if hasattr(result, 'pred_instances'):
        pred_instances = result.pred_instances
        
        scores = pred_instances.scores.cpu().numpy()
        valid_indices = scores > score_threshold
        
        if not np.any(valid_indices):
            return image, f"{title} (无检测结果)"
        
        bboxes = pred_instances.bboxes[valid_indices].cpu().numpy()
        labels = pred_instances.labels[valid_indices].cpu().numpy()
        scores = scores[valid_indices]
        
        vis_img = image.copy()
        
        for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
            x1, y1, x2, y2 = bbox.astype(int)
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            class_name = VOC_CLASSES_CN[label] if label < len(VOC_CLASSES_CN) else f"类别{label}"
            label_text = f"{class_name}: {score:.2f}"
            
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_img, (x1, y1-text_h-10), (x1+text_w, y1), (0, 255, 0), -1)
            cv2.putText(vis_img, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if hasattr(pred_instances, 'masks'):
            masks = pred_instances.masks[valid_indices].cpu().numpy()
            colors = plt.cm.tab10(np.linspace(0, 1, len(masks)))
            
            for mask, color in zip(masks, colors):
                color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                color_mask[mask] = (np.array(color[:3]) * 255).astype(np.uint8)
                
                vis_img = cv2.addWeighted(vis_img, 0.8, color_mask, 0.4, 0)
        
        info = f"{title} ({len(bboxes)} 个目标)"
        return vis_img, info
    
    return image, f"{title} (无检测结果)"

def visualize_comparison(image_path, models, save_dir="visualization_results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'检测结果对比 - {image_name}', fontsize=16, fontweight='bold')
    
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('原图', fontsize=12)
    axes[0, 0].axis('off')
    
    if 'mask_rcnn' in models:
        try:
            result = inference_detector(models['mask_rcnn'], image_path)
            vis_img, info = draw_bboxes_and_masks(image_rgb, result, "Mask R-CNN")
            axes[0, 1].imshow(vis_img)
            axes[0, 1].set_title(info, fontsize=12)
            axes[0, 1].axis('off')
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'Mask R-CNN\n推理失败:\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, 'Mask R-CNN\n模型未加载', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].axis('off')
    
    if 'sparse_rcnn' in models:
        try:
            result = inference_detector(models['sparse_rcnn'], image_path)
            vis_img, info = draw_bboxes_and_masks(image_rgb, result, "Sparse R-CNN")
            axes[1, 0].imshow(vis_img)
            axes[1, 0].set_title(info, fontsize=12)
            axes[1, 0].axis('off')
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Sparse R-CNN\n推理失败:\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'Sparse R-CNN\n模型未加载', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')
    
    axes[1, 1].text(0.5, 0.5, '性能比较\nMask R-CNN: 两阶段检测\nSparse R-CNN: 端到端检测\n\n详细性能数据请查看\n训练日志和测试结果', 
                   ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{image_name}_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存可视化结果: {save_path}")

def main():
    print("🎨 开始运行结果可视化...")
    
    models = load_models()
    
    if not models:
        print("❌ 没有成功加载任何模型")
        return
    
    test_images = get_test_images(4)
    
    if not test_images:
        print("❌ 没有找到测试图像")
        return
    
    for image_path in test_images:
        print(f"\n📸 处理图像: {os.path.basename(image_path)}")
        visualize_comparison(image_path, models)
    
    print("\n✅ 可视化完成！结果保存在 visualization_results/ 目录中")

if __name__ == "__main__":
    main() 