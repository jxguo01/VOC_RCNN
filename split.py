#!/usr/bin/env python3
# Dataset splitting script for VOC COCO format

import json
import random
import os

def split_dataset(input_file, train_ratio=0.8, val_ratio=0.1):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    random.shuffle(images)
    
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]
    
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}
    test_image_ids = {img['id'] for img in test_images}
    
    train_annotations = [ann for ann in data['annotations'] if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in data['annotations'] if ann['image_id'] in val_image_ids]
    test_annotations = [ann for ann in data['annotations'] if ann['image_id'] in test_image_ids]
    
    datasets = {
        'train': {'images': train_images, 'annotations': train_annotations},
        'val': {'images': val_images, 'annotations': val_annotations},
        'test': {'images': test_images, 'annotations': test_annotations}
    }
    
    base_dir = os.path.dirname(input_file)
    
    for split_name, split_data in datasets.items():
        output_data = {
            'images': split_data['images'],
            'annotations': split_data['annotations'],
            'categories': data['categories']
        }
        
        output_file = os.path.join(base_dir, f'instances_{split_name}2017.json')
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"{split_name}: {len(split_data['images'])} images, {len(split_data['annotations'])} annotations")

if __name__ == '__main__':
    random.seed(42)
    input_file = 'data/coco/annotations/instances_combined2017.json'
    split_dataset(input_file)