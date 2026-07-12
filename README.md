# Solar Filament Segmentation Challenge 2026

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)

An advanced deep learning pipeline for the automated, pixel-level segmentation of solar filaments in H-Alpha observations. This project aims to delineate complex solar structures, including fine-scale "barbs," while maintaining structural continuity and robustness across different solar observatories.

## Project Overview

Solar filaments are dense clouds of cool plasma suspended by magnetic fields above the solar surface. Their accurate segmentation is critical for predicting Coronal Mass Ejections (CMEs) and solar flares, which can significantly impact Earth's electric grids and satellite communications.

This project implements a state-of-the-art segmentation pipeline utilizing SegFormer and Mask2Former architectures, specifically optimized for high-resolution solar imagery (2048 x 2048 pixels) and handled under a ROCm/AMD GPU environment.

### Key Technical Highlights
- Adaptive Preprocessing: Implements a custom pipeline involving Otsu thresholding for disk masking and CLAHE (Contrast Limited Adaptive Histogram Equalization) to amplify subtle local contrast.
- Space-Aware Loss: A customized hybrid loss function (Weighted BCE + Weighted Dice) that ignores the "space" background, forcing the model to focus exclusively on the solar disk.
- High-Resolution Strategy: Utilizes a patching approach (512 x 512 patches) to preserve fine-scale structural details while managing VRAM constraints.
- Experiment Tracking: Fully integrated with MLflow for real-time monitoring of Dice scores, fragmentation metrics, and system hardware utilization.

---

## Project Structure

```text
.
├── dataset/                # Dataset handling and loading
│   ├── solar_dataset.py    # Custom PyTorch Dataset implementation
│   ├── mapper.py           # Data mapping utilities
│   └── __init__.py         # Package initialization
├── models/                 # Model architectures and configurations
│   ├── solar_segformer.py  # SegFormer-based implementation
│   ├── weighted_mask2former.py # Customized Mask2Former with disk weighting
│   └── model_config.py     # Hyperparameter configurations
├── utils/                  # Helper modules
│   ├── preprocessing.py    # Disk masking and CLAHE enhancement
│   └── coco_utils.py       # Polygon <-> Mask <-> RLE conversions
├── desc/                   # Project documentation and samples
│   ├── OVERVIEW.md         # High-level task description
│   └── DATA.md             # Detailed dataset specifications
├── tests/                  # Validation suite
│   └── test_dataset.py     # Dataset alignment and patch verification
├── train.py                # Main training and validation loop
├── requirements.txt        # Project dependencies
├── EDA_REPORT.md           # Exploratory Data Analysis findings
└── projects/               # External base implementations
    └── mask2former/        # Original Mask2Former repository
```

---

## Installation and Setup

### 1. Prerequisites
- Hardware: NVIDIA or AMD GPU (ROCm supported).
- OS: Linux (Ubuntu recommended).
- Python: 3.10+

### 2. Environment Setup
We recommend using a Conda environment:

```bash
# Create and activate environment
conda create -n solar_seg python=3.10
conda activate solar_seg

# Install dependencies
pip install -r requirements.txt
```

### 3. Dataset Preparation
The project expects the MAGFiLO dataset in the following structure:
```text
dataset/MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/
```
*Note: Update the DATA_DIR path in train.py to match your local filesystem.*

---

## Usage

### Training the Model
To start the training process with MLflow tracking enabled:

```bash
python train.py
```
The script will:
1. Load the SolarSegFormer (B0 backbone).
2. Apply preprocessing and patching on-the-fly.
3. Track metrics in mlflow.db.
4. Save the best weights to best_model.pth and state to checkpoint.pth.

### Monitoring Experiments
Launch the MLflow UI to visualize training curves and system metrics:

```bash
mlflow ui
```
Navigate to http://localhost:5000 to view the Solar_SegFormer_ROCm experiment.

### Running Tests
Verify that the image and mask alignment is correct across patches:

```bash
python tests/test_dataset.py
```

---

## Technical Architecture

### Preprocessing Pipeline
Solar images suffer from extremely low local contrast. Our pipeline solves this via:
1. Disk Masking: Uses Otsu thresholding and circle fitting to isolate the solar disk from the black space background.
2. Space Zeroing: All pixels outside the disk are set to 0, preventing the model from learning the trivial sun-space boundary.
3. CLAHE: Enhances the contrast of filament "barbs" relative to the solar disk, making subtle structures visible to the CNN/Transformer.

### Model Architectures
The project implements two primary approaches:

#### 1. SolarSegFormer
Based on the SegFormer architecture, utilizing a hierarchical Transformer encoder (mit-b0) and a lightweight All-MLP decoder.
- Optimized Loss: Loss = WeightedBCE(disk_mask) + WeightedDice(disk_mask).
- Resolution: Upsamples 1/4 resolution logits back to 2048 x 2048 using bilinear interpolation.

#### 2. Weighted Mask2Former
A customized version of Mask2Former that treats segmentation as a mask-classification problem.
- Innovation: We modified the SetCriterion to inject a disk-weighting mask into the Dice and Cross-Entropy loss components, ensuring that "space" pixels do not contribute to the gradient.

### Evaluation Metrics
- Dice Score: Measures the overlap between predicted and ground-truth masks.
- Fragmentation Ratio: (Number of Predicted Components) / (Number of GT Components). A value near 1.0 indicates high structural continuity.

---
Developed for the Solar Filament Segmentation Challenge 2026.
