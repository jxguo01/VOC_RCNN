# 基于VOC数据集的Mask R-CNN与Sparse R-CNN目标检测对比实验

本项目是神经网络与深度学习课程期中任务的完整实现，专注于在PASCAL VOC 2007数据集上训练和对比Mask R-CNN与Sparse R-CNN两种目标检测模型。

## 📋 项目概述

本实验实现并对比了两种先进的目标检测模型：
- **Mask R-CNN**: 经典两阶段检测器，具备实例分割能力
- **Sparse R-CNN**: 端到端检测器，使用稀疏目标查询
## 📦 模型权重下载

### 百度网盘下载
- **下载链接**: [https://pan.baidu.com/s/193-xZCt3r-ivDWYUJPCxXw?pwd=km8d]
- **文件大小**: 约3.5GB
- **包含内容**:
  - `mask_rcnn_epoch24.pth` (337MB) - **推荐使用**
  - `sparse_rcnn_epoch12.pth` (1.9GB) - 基础版本
  - `sparse_rcnn_improved_epoch36.pth` (1.2GB) - 改进版本

### 使用说明
```python
# 下载后放置在项目根目录，然后加载模型
from mmdet.apis import init_detector

# 推荐使用 Mask R-CNN
config_file = 'mask_rcnn_low_memory.py'
checkpoint_file = 'mask_rcnn_epoch24.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
```

## 🚀 快速开始

### 环境要求
```bash
# 核心依赖包
Python 3.9.21
torch 2.1.0+cu121
mmdet 3.3.0
mmcv 2.1.0
mmengine 0.10.1

# 安装命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmdet==3.3.0
pip install matplotlib opencv-python numpy pillow
```

### 数据准备
```bash
# 1. 下载VOC 2007数据集
# 2. 转换为COCO格式并生成分割掩码
python convert_voc_to_coco.py

# 3. 数据集分割
python split.py
```

### 模型训练

#### 1. 训练Mask R-CNN (内存优化版)
```bash
python tools/train.py mask_rcnn_low_memory.py
```

#### 2. 训练Sparse R-CNN (内存优化版)
```bash
python tools/train.py sparse_rcnn_low_memory.py
```

### 模型测试与可视化

#### 1. 生成综合对比分析
```bash
python test_and_visualize_models.py
```

#### 2. 外部图像测试
```bash
python test_external_images.py
```

#### 3. 训练曲线分析
```bash
python generate_tensorboard_plots.py
```
### 训练效率对比

| 指标 | Mask R-CNN | Sparse R-CNN | 优势 |
|------|------------|--------------|------|
| 训练时间 | 53分钟 | 45分钟 | Sparse R-CNN快15% |
| 内存使用 | 1.2-1.3GB | ~3.5GB | Mask R-CNN省73% |
| 损失收敛 | 6.8→3.2 (52%↓) | 75.5→54.9 (27%↓) | Mask R-CNN更稳定 |
| 模型大小 | 281MB | 1.9GB | Mask R-CNN小85% |

### 外部图像泛化测试

| 场景类型 | 预期目标 | Mask R-CNN检测数 | Sparse R-CNN检测数 | 检测优势 |
|---------|---------|------------------|-------------------|----------|
| 街景图像 | 汽车、人、自行车 | 4 | 2 | +100% |
| 室内场景 | 沙发、椅子、电视 | 3 | 1 | +200% |
| 动物场景 | 马、狗、鸟 | 5 | 3 | +67% |
| **总计** | **多类别混合** | **12** | **6** | **+100%** |

## 📁 项目结构

```
VOC_RCNN2/
├── 配置文件/
│   ├── mask_rcnn_low_memory.py              # Mask R-CNN内存优化配置
│   ├── sparse_rcnn_low_memory.py            # Sparse R-CNN内存优化配置
│   ├── mask-rcnn_r50_fpn_ms-poly-2x_voc.py # 标准Mask R-CNN配置
│   └── sparse-rcnn_r50_fpn_1x_voc.py        # 标准Sparse R-CNN配置
├── 数据处理/
│   ├── convert_voc_to_coco.py               # VOC到COCO格式转换
│   ├── split.py                             # 数据集分割
│   └── coco.py                              # COCO数据集配置
├── 训练和测试/
│   ├── test_and_visualize_models.py         # 模型测试与可视化
│   ├── test_external_images.py              # 外部图像测试
│   ├── generate_tensorboard_plots.py        # 训练曲线生成
│   └── visualize_test_images.py             # 测试图像可视化
├── 数据集/
│   └── data/coco/                           # COCO格式VOC数据集
├── 模型权重/
│   └── work_dirs/                           # 训练输出和模型权重
├── 结果文件/
│   ├── visualization_results/               # 可视化结果
│   ├── tensorboard_plots/                   # 训练曲线图
│   ├── test_image_visualizations/           # 测试图像对比
│   └── external_image_tests/                # 外部图像测试结果
└── 报告文档/
    └── report_latest.pdf # 最新版实验报告
```

## 🎨 生成的可视化内容

### 1. 训练过程分析
- **文件位置**: `tensorboard_plots/training_curves.png`
- **内容**: 两模型的损失曲线和mAP进展对比
- **生成脚本**: `generate_tensorboard_plots.py`

### 2. VOC测试集检测结果
- **文件位置**: `test_image_visualizations/*.png`
- **内容**: 4面板对比展示：
  - 原始图像
  - Mask R-CNN检测结果
  - Sparse R-CNN检测结果
  - 性能对比分析
- **生成脚本**: `visualize_test_images.py`

### 3. 外部图像泛化测试
- **文件位置**: `external_image_tests/*.png`
- **内容**: 3种外部场景的模型对比：
  - 街景 (汽车、行人、自行车)
  - 室内 (家具、电器、物品)
  - 户外动物 (马、牛、鸟、狗)
- **生成脚本**: `test_external_images.py`

### 4. 综合性能分析
- **文件位置**: `visualization_results/detailed_metrics.png`
- **内容**: 详细的性能指标可视化对比
- **生成脚本**: `test_and_visualize_models.py`

## 🔧 关键技术实现

### 内存优化策略

#### Mask R-CNN优化
```python
# 输入尺寸优化
input_size = (800, 600)  # 从(1333,800)优化
batch_size = 1           # 从2优化
max_per_img = 100        # RPN proposals从2000优化

# 内存节省效果：4.6GB → 1.3GB (75%节省)
```

#### Sparse R-CNN优化
```python
# 模型参数优化
num_proposals = 100      # 从300优化
ffn_channels = 1024      # 从2048优化
batch_size = 1           # 单GPU训练

# 内存使用：约3.5GB
```

### 数据处理创新
- **掩码提取**: 使用OpenCV轮廓检测从二值掩码生成多边形格式
- **格式转换**: 完整的VOC到COCO格式转换，保留分割信息
- **数据增强**: 针对小数据集的适配策略

## 📈 核心发现

### 性能对比分析
1. **检测精度**: Mask R-CNN的mAP@0.5:0.95是Sparse R-CNN的5.7倍
2. **训练效率**: Mask R-CNN收敛更快更稳定
3. **内存效率**: Mask R-CNN内存使用仅为Sparse R-CNN的37%
4. **泛化能力**: Mask R-CNN在外部图像上检测数量是Sparse R-CNN的2倍

### 技术架构优势
1. **两阶段 vs 端到端**: 在中等规模数据集上，两阶段方法显著优于端到端方法
2. **RPN + ROI设计**: 传统的区域提议+精细化策略更有效
3. **稀疏查询限制**: 100个固定查询可能不足以覆盖复杂场景

## 🛠️ 复现指南

### 环境配置
```bash
# 1. 创建虚拟环境
conda create -n voc_rcnn python=3.9
conda activate voc_rcnn

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证环境
python -c "import mmdet; print(mmdet.__version__)"
```

### 训练复现
```bash
# 1. 数据准备
python convert_voc_to_coco.py
python split.py

# 2. 训练Mask R-CNN
python tools/train.py mask_rcnn_low_memory.py

# 3. 训练Sparse R-CNN  
python tools/train.py sparse_rcnn_low_memory.py

# 4. 生成所有结果
python test_and_visualize_models.py
python test_external_images.py
python generate_tensorboard_plots.py
```

### 模型权重
- **Mask R-CNN**: `work_dirs/mask_rcnn_restart_clean/epoch_12.pth` (281MB)
- **Sparse R-CNN**: `work_dirs/sparse_rcnn_improved_final/epoch_12.pth` (1.9GB)

## 📚 技术栈

### 框架与工具
- **MMDetection 3.3.0**: 核心目标检测框架
- **MMEngine**: 训练和推理引擎  
- **PyTorch 2.1.0**: 深度学习后端
- **CUDA 12.1**: GPU加速
- **OpenCV**: 图像处理
- **Matplotlib**: 可视化

### 硬件环境
- **GPU**: RTX 2060 6GB
- **Python**: 3.9.21

## 🎯 实验结论

### 主要发现
1. **两阶段检测器优势明显**: 在VOC数据集上，Mask R-CNN性能显著优于Sparse R-CNN
2. **内存优化成功**: 成功将Mask R-CNN内存使用从4.6GB降至1.3GB
3. **训练效率**: Mask R-CNN训练更稳定，收敛更快
4. **泛化能力**: 两模型均能在外部图像上检测VOC类别物体，但Mask R-CNN效果更好

### 实践建议
- **VOC规模数据集**: 推荐使用Mask R-CNN
- **资源受限环境**: Mask R-CNN提供更好的性能/成本比
- **研究用途**: Sparse R-CNN适合作为端到端检测的基线模型

### 🔗 项目链接
- **模型权重 (百度网盘)**: [https://pan.baidu.com/s/193-xZCt3r-ivDWYUJPCxXw?pwd=km8d]

### 📋 完整文档
- **最新实验报告 (PDF)**: `report_latest.pdf`

