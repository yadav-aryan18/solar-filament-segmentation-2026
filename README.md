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
│   ├── solar_dataset.py    # Custom PyTorch Dataset (dynamic train / static val patching)
│   ├── mapper.py           # Data mapping utilities
│   └── __init__.py         # Package initialization
├── models/                 # Model architectures and configurations
│   ├── solar_segformer.py  # SegFormer-based implementation
│   ├── weighted_mask2former.py # Customized Mask2Former with disk weighting
│   └── model_config.py     # Hyperparameter configurations
├── utils/                  # Helper modules
│   ├── preprocessing.py    # Disk masking and CLAHE enhancement
│   ├── coco_utils.py       # Polygon <-> Mask <-> RLE conversions
│   └── postprocess.py      # Morphological cleanup + connected-component filtering
├── desc/                   # Project documentation and samples
│   ├── OVERVIEW.md         # High-level task description
│   └── DATA.md             # Detailed dataset specifications
├── tests/                  # Validation suite
│   ├── test_dataset.py         # Dataset alignment and patch verification
│   ├── test_dataset_pipeline.py # End-to-end preprocessing -> patch -> tensor checks
│   └── test_csv_format.py      # RLE encoding / submission CSV format checks
├── train.py                # Main training and validation loop
├── inference.py            # Tiled sliding-window inference + submission generation
├── requirements.txt        # Project dependencies
└── EDA_REPORT.md           # Exploratory Data Analysis findings
```

> Note: `projects/mask2former/` holds a local clone of the upstream Mask2Former
> repository for reference; it is intentionally not tracked by git, and the
> downloaded dataset archive (`dataset/*.zip`) is likewise excluded. See
> `.gitignore`.

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
Place the downloaded MAGFiLO archive (`filament-segmentation-2026.zip`) under
`dataset/` and extract it so the structure matches:
```text
dataset/MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/
```
The archive itself is not tracked by git (see `.gitignore`); only the extracted
directory is used locally. *Update the `DATA_DIR` / `JSON_PATH` / `IMG_DIR`
paths in `train.py` and `inference.py` to match your local filesystem.*

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

### Running Inference and Generating a Submission
`inference.py` reconstructs the full-resolution model predictions using a tiled
sliding-window strategy with Hann-window blending (to avoid hard seams between
patches), then passes the stitched mask through morphological cleanup before
encoding it as RLE rows for the challenge submission CSV.

```bash
python inference.py
```
The script will:
1. Load the trained `best_model.pth` (add patch size / stride / overlap via
   `TiledInference` defaults if not overridden).
2. Process each test image with overlapping 512x512 tiles, blending outputs in
   the overlap regions.
3. Apply `utils/postprocess.py` (small-component removal, gap filling) to the
   stitched mask.
4. Encode per-image masks as RLE via `coco_utils.mask_to_rle` and write
   `submission.csv` in the competition format.

### Running Tests
The `tests/` suite checks dataset alignment, the full preprocessing -> patching
pipeline, and the RLE / submission-CSV format:

```bash
python tests/test_dataset.py
python tests/test_dataset_pipeline.py
python tests/test_csv_format.py
```

---

## Technical Architecture

### Preprocessing Pipeline
Solar images suffer from extremely low local contrast. Our pipeline solves this via:
1. Disk Masking: Uses Otsu thresholding and circle fitting to isolate the solar disk from the black space background.
2. Space Zeroing: All pixels outside the disk are set to 0, preventing the model from learning the trivial sun-space boundary.
3. CLAHE: Enhances the contrast of filament "barbs" relative to the solar disk, making subtle structures visible to the CNN/Transformer.

### Patching Strategy
`dataset/solar_dataset.py` supports two patching modes selected by the `is_val`
flag (set from `train.py` for the train vs. validation split):
- **Training (`is_val=False`):** *dynamic* patching — random valid patches are
  resampled per image each epoch for data augmentation, and `__len__` scales with
  `num_images * patches_per_image`.
- **Validation (`is_val=True`):** *static* patching — a fixed list of patches is
  precomputed once (`_generate_static_patch_list`) so the validation metric is
  computed over a consistent, reproducible patch set across epochs.

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

### Postprocessing
`utils/postprocess.py` refines raw model output into submission-ready masks:
- **Small-component removal:** drops connected components below a minimum pixel
  size to suppress speckle noise.
- **Morphological cleanup:** fills small gaps and holes to preserve filament
  structural continuity (reducing the fragmentation ratio).
These steps are composed in `full_postprocess()` and invoked by `inference.py`
after the tiled mask is stitched, before RLE encoding the final predictions.

---
Developed for the Solar Filament Segmentation Challenge 2026.
