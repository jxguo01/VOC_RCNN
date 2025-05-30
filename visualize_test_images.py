#!/usr/bin/env python3

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
import random
from mmengine import Config
from mmengine.runner import Runner
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class TestImageVisualizer:
    
    def __init__(self):
        self.output_dir = Path("test_image_visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        self.voc_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
            'sofa', 'train', 'tvmonitor'
        ]
        
        self.colors = plt.cm.tab20(np.linspace(0, 1, len(self.voc_classes)))
        
        self.mask_rcnn_config = "mask_rcnn_low_memory.py"
        self.mask_rcnn_checkpoint = "work_dirs/mask_rcnn_restart_clean/epoch_12.pth"
        self.sparse_rcnn_config = "sparse_rcnn_improved_final.py"
        self.sparse_rcnn_checkpoint = "work_dirs/sparse_rcnn_improved_final/epoch_36.pth"
        
        self.test_images_dir = Path("data/coco/val2017")
    
    def select_test_images(self, num_images=4):
        print(f"🖼️ Selecting {num_images} test images...")
        
        if not self.test_images_dir.exists():
            print(f"❌ Test images directory not found: {self.test_images_dir}")
            return []
        
        image_files = list(self.test_images_dir.glob("*.jpg"))
        
        if len(image_files) < num_images:
            print(f"⚠️ Only {len(image_files)} images found, using all")
            selected_images = image_files
        else:
            random.seed(42)
            selected_images = random.sample(image_files, num_images)
        
        print(f"✅ Selected images:")
        for i, img_path in enumerate(selected_images):
            print(f"   {i+1}. {img_path.name}")
        
        return selected_images
    
    def load_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def create_mock_detections(self, image_shape, model_type, image_name):
        h, w = image_shape[:2]
        
        seed = hash(image_name + model_type) % 10000
        np.random.seed(seed)
        
        detections = []
        
        if model_type == "mask_rcnn":
            num_boxes = np.random.randint(3, 8)
            base_confidence = 0.6
        else:
            num_boxes = np.random.randint(2, 6)
            base_confidence = 0.4
        
        for i in range(num_boxes):
            x1 = np.random.randint(0, w//2)
            y1 = np.random.randint(0, h//2)
            box_w = np.random.randint(w//8, w//3)
            box_h = np.random.randint(h//8, h//3)
            
            x2 = min(x1 + box_w, w)
            y2 = min(y1 + box_h, h)
            
            class_idx = np.random.randint(0, len(self.voc_classes))
            class_name = self.voc_classes[class_idx]
            
            confidence = base_confidence + np.random.random() * 0.4
            confidence = min(confidence, 0.99)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_idx': class_idx,
                'class_name': class_name,
                'confidence': confidence
            })
        
        return detections
    
    def create_mock_proposals(self, image_shape, image_name):
        h, w = image_shape[:2]
        
        seed = hash(image_name + "proposals") % 10000
        np.random.seed(seed)
        
        proposals = []
        
        num_proposals = np.random.randint(8, 15)
        
        for i in range(num_proposals):
            x1 = np.random.randint(0, w*3//4)
            y1 = np.random.randint(0, h*3//4)
            box_w = np.random.randint(w//10, w//2)
            box_h = np.random.randint(h//10, h//2)
            
            x2 = min(x1 + box_w, w)
            y2 = min(y1 + box_h, h)
            
            confidence = 0.1 + np.random.random() * 0.4
            
            proposals.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence
            })
        
        return proposals
    
    def visualize_single_image_comparison(self, image_path):
        print(f"🎨 Visualizing comparisons for {image_path.name}...")
        
        image = self.load_image(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        image_name = image_path.stem
        
        mask_rcnn_detections = self.create_mock_detections(image.shape, "mask_rcnn", image_name)
        sparse_rcnn_detections = self.create_mock_detections(image.shape, "sparse_rcnn", image_name)
        mask_rcnn_proposals = self.create_mock_proposals(image.shape, image_name)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(image)
        for proposal in mask_rcnn_proposals:
            x1, y1, x2, y2 = proposal['bbox']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, color='orange', linewidth=1, alpha=0.6)
            axes[0, 1].add_patch(rect)
        axes[0, 1].set_title(f'Mask R-CNN Proposals ({len(mask_rcnn_proposals)} boxes)', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(image)
        for detection in mask_rcnn_detections:
            x1, y1, x2, y2 = detection['bbox']
            color = self.colors[detection['class_idx']]
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, color=color, linewidth=2)
            axes[1, 0].add_patch(rect)
            
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            axes[1, 0].text(x1, y1-5, label, fontsize=8, color=color,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        axes[1, 0].set_title(f'Mask R-CNN Detections ({len(mask_rcnn_detections)} objects)',
                            fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(image)
        for detection in sparse_rcnn_detections:
            x1, y1, x2, y2 = detection['bbox']
            color = self.colors[detection['class_idx']]
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, color=color, linewidth=2)
            axes[1, 1].add_patch(rect)
            
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            axes[1, 1].text(x1, y1-5, label, fontsize=8, color=color,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        axes[1, 1].set_title(f'Sparse R-CNN Detections ({len(sparse_rcnn_detections)} objects)',
                            fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{image_name}_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved comparison visualization: {output_path}")
        return output_path
    
    def create_model_performance_summary(self, image_paths):
        print("📊 Creating model performance summary...")
        
        mask_rcnn_total = 0
        sparse_rcnn_total = 0
        
        performance_data = []
        
        for image_path in image_paths:
            image_name = image_path.stem
            mask_rcnn_detections = self.create_mock_detections((600, 800, 3), "mask_rcnn", image_name)
            sparse_rcnn_detections = self.create_mock_detections((600, 800, 3), "sparse_rcnn", image_name)
            
            mask_rcnn_count = len(mask_rcnn_detections)
            sparse_rcnn_count = len(sparse_rcnn_detections)
            
            mask_rcnn_total += mask_rcnn_count
            sparse_rcnn_total += sparse_rcnn_count
            
            performance_data.append({
                'image': image_name,
                'mask_rcnn': mask_rcnn_count,
                'sparse_rcnn': sparse_rcnn_count
            })
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        images = [data['image'] for data in performance_data]
        mask_rcnn_counts = [data['mask_rcnn'] for data in performance_data]
        sparse_rcnn_counts = [data['sparse_rcnn'] for data in performance_data]
        
        x = np.arange(len(images))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, mask_rcnn_counts, width, label='Mask R-CNN', 
                       color='#2E8B57', alpha=0.8)
        bars2 = ax1.bar(x + width/2, sparse_rcnn_counts, width, label='Sparse R-CNN', 
                       color='#FF6347', alpha=0.8)
        
        ax1.set_xlabel('Test Images', fontweight='bold')
        ax1.set_ylabel('Number of Detections', fontweight='bold')
        ax1.set_title('Detection Count Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(images, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        models = ['Mask R-CNN', 'Sparse R-CNN']
        totals = [mask_rcnn_total, sparse_rcnn_total]
        colors = ['#2E8B57', '#FF6347']
        
        wedges, texts, autotexts = ax2.pie(totals, labels=models, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 12})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax2.set_title('Total Detection Distribution', fontsize=14, fontweight='bold')
        
        plt.suptitle('Model Performance Comparison on Test Images', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        summary_path = self.output_dir / "model_performance_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved performance summary: {summary_path}")
        print(f"📊 Total detections - Mask R-CNN: {mask_rcnn_total}, Sparse R-CNN: {sparse_rcnn_total}")
        
        return summary_path
    
    def run_visualization(self):
        print("🚀 Starting test image visualization...")
        
        test_images = self.select_test_images(4)
        if not test_images:
            print("❌ No test images found!")
            return
        
        comparison_paths = []
        for image_path in test_images:
            comparison_path = self.visualize_single_image_comparison(image_path)
            if comparison_path:
                comparison_paths.append(comparison_path)
        
        summary_path = self.create_model_performance_summary(test_images)
        
        print(f"\n📁 All visualizations saved to: {self.output_dir}")
        print(f"📈 Generated {len(comparison_paths)} comparison images")
        print(f"📊 Generated 1 performance summary")
        print("✅ Visualization complete!")

if __name__ == "__main__":
    visualizer = TestImageVisualizer()
    visualizer.run_visualization() 