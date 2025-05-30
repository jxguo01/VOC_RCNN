#!/usr/bin/env python3
# VOC to COCO format converter with segmentation masks

import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from pycocotools import mask as maskUtils
import numpy as np
import cv2

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def load_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotation = {}
    annotation['filename'] = root.find('filename').text
    
    size = root.find('size')
    annotation['width'] = int(size.find('width').text)
    annotation['height'] = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in VOC_CLASSES:
            continue
            
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    annotation['objects'] = objects
    return annotation

def create_mask_from_bbox(bbox, img_width, img_height):
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    xmin, ymin, xmax, ymax = map(int, bbox)
    mask[ymin:ymax, xmin:xmax] = 1
    return mask

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)
    
    return polygons

def convert_voc_to_coco(voc_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    
    categories = []
    for i, cls_name in enumerate(VOC_CLASSES):
        categories.append({
            'id': i + 1,
            'name': cls_name,
            'supercategory': 'object'
        })
    
    splits = ['train', 'val']
    
    for split in splits:
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': categories
        }
        
        image_id = 0
        annotation_id = 0
        
        split_file = os.path.join(voc_dir, 'ImageSets', 'Main', f'{split}.txt')
        if not os.path.exists(split_file):
            print(f"Split file not found: {split_file}")
            continue
            
        with open(split_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]
        
        for image_name in image_names:
            image_id += 1
            
            img_path = os.path.join(voc_dir, 'JPEGImages', f'{image_name}.jpg')
            xml_path = os.path.join(voc_dir, 'Annotations', f'{image_name}.xml')
            
            if not os.path.exists(img_path) or not os.path.exists(xml_path):
                continue
            
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            coco_data['images'].append({
                'id': image_id,
                'file_name': f'{image_name}.jpg',
                'width': img_width,
                'height': img_height
            })
            
            annotation = load_voc_annotation(xml_path)
            
            for obj in annotation['objects']:
                annotation_id += 1
                category_id = VOC_CLASSES.index(obj['name']) + 1
                
                bbox = obj['bbox']
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                area = w * h
                
                mask = create_mask_from_bbox(bbox, img_width, img_height)
                polygons = mask_to_polygon(mask)
                
                if polygons:
                    segmentation = polygons
                    rle = maskUtils.encode(np.asfortranarray(mask))
                    rle['counts'] = rle['counts'].decode('utf-8')
                else:
                    segmentation = []
                    rle = None
                
                coco_annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [x, y, w, h],
                    'area': area,
                    'segmentation': segmentation,
                    'iscrowd': 0
                }
                
                coco_data['annotations'].append(coco_annotation)
        
        output_file = os.path.join(output_dir, 'annotations', f'instances_{split}2017.json')
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Converted {split} set: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")

if __name__ == '__main__':
    voc_dir = 'data/VOCdevkit/VOC2007'
    output_dir = 'data/coco'
    convert_voc_to_coco(voc_dir, output_dir) 