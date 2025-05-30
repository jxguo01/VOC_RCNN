#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.getcwd())

from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmengine.structures import InstanceData

class ExternalImageTester:
    def __init__(self):
        self.mask_rcnn_config = "mask_rcnn.py"
        self.mask_rcnn_checkpoint = "work_dirs/mask_rcnn/epoch_24.pth"
        self.sparse_rcnn_config = "sparse_rcnn.py"
        self.sparse_rcnn_checkpoint = "work_dirs/sparse_rcnn/epoch_12.pth"
        
        self.external_images_dir = Path("external_images")
        self.results_dir = Path("external_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.voc_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
            'sofa', 'train', 'tvmonitor'
        ]
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.models = {}
        
    def load_models(self):
        print("Loading models...")
        
        try:
            print("Loading Mask R-CNN...")
            if os.path.exists(self.mask_rcnn_checkpoint):
                self.models['mask_rcnn'] = init_detector(
                    self.mask_rcnn_config, 
                    self.mask_rcnn_checkpoint, 
                    device=self.device
                )
                print("✅ Mask R-CNN loaded")
            else:
                print(f"❌ Mask R-CNN checkpoint not found: {self.mask_rcnn_checkpoint}")
                
        except Exception as e:
            print(f"❌ Failed to load Mask R-CNN: {e}")
            
        try:
            print("Loading Sparse R-CNN...")
            if os.path.exists(self.sparse_rcnn_checkpoint):
                self.models['sparse_rcnn'] = init_detector(
                    self.sparse_rcnn_config, 
                    self.sparse_rcnn_checkpoint, 
                    device=self.device
                )
                print("✅ Sparse R-CNN loaded")
            else:
                print(f"❌ Sparse R-CNN checkpoint not found: {self.sparse_rcnn_checkpoint}")
                
        except Exception as e:
            print(f"❌ Failed to load Sparse R-CNN: {e}")
        
        print(f"Total loaded models: {len(self.models)}")
        
    def download_sample_images(self):
        print("Creating sample external images...")
        
        if not self.external_images_dir.exists():
            self.external_images_dir.mkdir()
        
        sample_images = [
            ("street_scene.jpg", self.create_sample_street_scene),
            ("indoor_scene.jpg", self.create_sample_indoor_scene),
            ("animal_scene.jpg", self.create_sample_animal_scene)
        ]
        
        for filename, creator_func in sample_images:
            image_path = self.external_images_dir / filename
            if not image_path.exists():
                creator_func(image_path)
                print(f"✅ Created: {filename}")
        
        return [self.external_images_dir / name for name, _ in sample_images]
    
    def create_sample_street_scene(self, output_path):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        cv2.rectangle(img, (50, 300), (200, 400), (100, 100, 255), -1)
        cv2.rectangle(img, (60, 320), (80, 380), (255, 255, 255), -1)
        cv2.rectangle(img, (120, 320), (140, 380), (255, 255, 255), -1)
        cv2.rectangle(img, (80, 320), (120, 350), (200, 200, 200), -1)
        
        cv2.rectangle(img, (400, 280), (600, 420), (50, 50, 200), -1)
        cv2.rectangle(img, (420, 300), (450, 340), (255, 255, 255), -1)
        cv2.rectangle(img, (520, 300), (550, 340), (255, 255, 255), -1)
        cv2.rectangle(img, (450, 300), (520, 320), (150, 150, 150), -1)
        
        cv2.circle(img, (100, 400), 20, (0, 0, 0), -1)
        cv2.circle(img, (150, 400), 20, (0, 0, 0), -1)
        cv2.circle(img, (450, 420), 25, (0, 0, 0), -1)
        cv2.circle(img, (550, 420), 25, (0, 0, 0), -1)
        
        cv2.rectangle(img, (0, 430), (640, 480), (100, 100, 100), -1)
        
        cv2.imwrite(str(output_path), img)
    
    def create_sample_indoor_scene(self, output_path):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 220
        
        cv2.rectangle(img, (200, 200), (450, 320), (139, 69, 19), -1)
        cv2.rectangle(img, (210, 180), (440, 200), (101, 67, 33), -1)
        
        cv2.rectangle(img, (220, 300), (270, 320), (160, 82, 45), -1)
        cv2.rectangle(img, (280, 300), (330, 320), (160, 82, 45), -1)
        cv2.rectangle(img, (340, 300), (390, 320), (160, 82, 45), -1)
        cv2.rectangle(img, (400, 300), (430, 320), (160, 82, 45), -1)
        
        cv2.rectangle(img, (100, 350), (150, 450), (160, 82, 45), -1)
        cv2.rectangle(img, (120, 330), (130, 350), (101, 67, 33), -1)
        cv2.rectangle(img, (130, 350), (140, 380), (139, 69, 19), -1)
        
        cv2.rectangle(img, (500, 100), (620, 350), (128, 0, 128), -1)
        cv2.rectangle(img, (520, 120), (560, 160), (255, 255, 255), -1)
        
        cv2.rectangle(img, (0, 460), (640, 480), (101, 67, 33), -1)
        
        cv2.imwrite(str(output_path), img)
    
    def create_sample_animal_scene(self, output_path):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 180
        img[:240, :] = [135, 206, 235]
        img[240:, :] = [34, 139, 34]
        
        cv2.ellipse(img, (200, 320), (60, 40), 0, 0, 360, (139, 69, 19), -1)
        cv2.ellipse(img, (170, 300), (25, 20), 0, 0, 360, (139, 69, 19), -1)
        cv2.circle(img, (160, 295), 3, (0, 0, 0), -1)
        cv2.ellipse(img, (150, 295), (8, 5), 0, 0, 360, (0, 0, 0), -1)
        cv2.rectangle(img, (180, 350), (190, 380), (139, 69, 19), -1)
        cv2.rectangle(img, (210, 350), (220, 380), (139, 69, 19), -1)
        cv2.rectangle(img, (190, 340), (210, 360), (139, 69, 19), -1)
        
        cv2.ellipse(img, (450, 340), (70, 45), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (420, 315), (30, 25), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (410, 310), 3, (0, 0, 0), -1)
        cv2.ellipse(img, (400, 310), (8, 5), 0, 0, 360, (255, 192, 203), -1)
        cv2.rectangle(img, (430, 375), (440, 400), (255, 255, 255), -1)
        cv2.rectangle(img, (460, 375), (470, 400), (255, 255, 255), -1)
        
        cv2.circle(img, (100, 100), 30, (255, 255, 0), -1)
        
        cv2.imwrite(str(output_path), img)
        
    def test_model_on_image(self, model, model_name, image_path):
        print(f"Testing {model_name} on {image_path.name}...")
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
            
        try:
            result = inference_detector(model, image)
            
            detections = []
            pred_instances = result.pred_instances
            
            if len(pred_instances.bboxes) > 0:
                scores = pred_instances.scores.cpu().numpy()
                labels = pred_instances.labels.cpu().numpy()
                bboxes = pred_instances.bboxes.cpu().numpy()
                
                for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
                    if score > 0.3:
                        detections.append({
                            'bbox': bbox.tolist(),
                            'score': float(score),
                            'label': int(label),
                            'class_name': self.voc_classes[label]
                        })
            
            print(f"✅ {model_name}: {len(detections)} detections")
            return detections
            
        except Exception as e:
            print(f"❌ Error during inference with {model_name}: {e}")
            return None
    
    def visualize_detections(self, image_path, mask_rcnn_detections, sparse_rcnn_detections):
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(image_rgb)
        if mask_rcnn_detections:
            for det in mask_rcnn_detections:
                x1, y1, x2, y2 = det['bbox']
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, color='red', linewidth=2)
                axes[1].add_patch(rect)
                axes[1].text(x1, y1-5, f"{det['class_name']}: {det['score']:.2f}",
                           color='red', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        axes[1].set_title(f'Mask R-CNN ({len(mask_rcnn_detections or [])} detections)', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(image_rgb)
        if sparse_rcnn_detections:
            for det in sparse_rcnn_detections:
                x1, y1, x2, y2 = det['bbox']
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, color='blue', linewidth=2)
                axes[2].add_patch(rect)
                axes[2].text(x1, y1-5, f"{det['class_name']}: {det['score']:.2f}",
                           color='blue', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        axes[2].set_title(f'Sparse R-CNN ({len(sparse_rcnn_detections or [])} detections)', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        output_path = self.results_dir / f"{image_path.stem}_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved visualization: {output_path}")
        return output_path
    
    def create_summary_report(self, test_results):
        print("Creating summary report...")
        
        total_mask_rcnn = sum(len(result['mask_rcnn'] or []) for result in test_results.values())
        total_sparse_rcnn = sum(len(result['sparse_rcnn'] or []) for result in test_results.values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        images = list(test_results.keys())
        mask_rcnn_counts = [len(test_results[img]['mask_rcnn'] or []) for img in images]
        sparse_rcnn_counts = [len(test_results[img]['sparse_rcnn'] or []) for img in images]
        
        x = np.arange(len(images))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, mask_rcnn_counts, width, label='Mask R-CNN', 
                       color='red', alpha=0.7)
        bars2 = ax1.bar(x + width/2, sparse_rcnn_counts, width, label='Sparse R-CNN', 
                       color='blue', alpha=0.7)
        
        ax1.set_xlabel('Test Images')
        ax1.set_ylabel('Number of Detections')
        ax1.set_title('Detection Count Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([img.replace('_scene', '') for img in images], rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        models = ['Mask R-CNN', 'Sparse R-CNN']
        totals = [total_mask_rcnn, total_sparse_rcnn]
        colors = ['red', 'blue']
        
        wedges, texts, autotexts = ax2.pie(totals, labels=models, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title('Total Detection Distribution')
        
        plt.tight_layout()
        summary_path = self.results_dir / "external_test_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Summary report saved: {summary_path}")
        print(f"Total detections - Mask R-CNN: {total_mask_rcnn}, Sparse R-CNN: {total_sparse_rcnn}")
        
        return summary_path
    
    def run_tests(self):
        print("🚀 Starting external image testing...")
        
        self.load_models()
        
        if not self.models:
            print("❌ No models loaded successfully!")
            return
        
        image_paths = self.download_sample_images()
        
        test_results = {}
        visualization_paths = []
        
        for image_path in image_paths:
            image_name = image_path.stem
            print(f"\n📸 Testing image: {image_name}")
            
            mask_rcnn_results = None
            sparse_rcnn_results = None
            
            if 'mask_rcnn' in self.models:
                mask_rcnn_results = self.test_model_on_image(
                    self.models['mask_rcnn'], 'Mask R-CNN', image_path)
            
            if 'sparse_rcnn' in self.models:
                sparse_rcnn_results = self.test_model_on_image(
                    self.models['sparse_rcnn'], 'Sparse R-CNN', image_path)
            
            test_results[image_name] = {
                'mask_rcnn': mask_rcnn_results,
                'sparse_rcnn': sparse_rcnn_results
            }
            
            viz_path = self.visualize_detections(image_path, mask_rcnn_results, sparse_rcnn_results)
            visualization_paths.append(viz_path)
        
        summary_path = self.create_summary_report(test_results)
        
        results_json_path = self.results_dir / "test_results.json"
        with open(results_json_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n📁 All results saved to: {self.results_dir}")
        print(f"📊 Generated {len(visualization_paths)} comparison images")
        print(f"📈 Generated summary report")
        print(f"💾 Saved detailed results to JSON")
        print("✅ External image testing complete!")

if __name__ == "__main__":
    tester = ExternalImageTester()
    tester.run_tests() 