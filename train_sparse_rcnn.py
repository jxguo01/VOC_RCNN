#!/usr/bin/env python3
# Sparse R-CNN training script

import os
import subprocess
import sys

def train_sparse_rcnn():
    config_file = 'sparse_rcnn.py'
    
    if not os.path.exists(config_file):
        print(f"Error: Config file {config_file} not found")
        return False
    
    cmd = [
        sys.executable, '-m', 'mmdet.tools.train',
        config_file,
        '--work-dir', 'work_dirs/sparse_rcnn_training'
    ]
    
    print(f"Starting Sparse R-CNN training...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == '__main__':
    success = train_sparse_rcnn()
    
    if success:
        print("✅ Sparse R-CNN training completed successfully")
    else:
        print("❌ Sparse R-CNN training failed")
        sys.exit(1) 