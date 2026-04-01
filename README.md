# RxRx1 siRNA Classification

Kaggle competition: Recursion Cellular Image Classification

## Problem

Given microscopy images of cells treated with siRNAs (genetic perturbations), predict which siRNA was applied to each well. The challenge is that replicates of the same treatment do not look identical due to experimental noise.

## Data

- 51 experimental batches
- 4 plates per batch, 308 wells per plate
- 2 imaging sites per well
- 6 channels per site

## Setup

```bash
pip install -r requirements.txt
```

## Running

Copy any of the approach files to your Kaggle notebook:

| File | Model | Sites |
|------|-------|-------|
| 01_resnet50_single_site.py | ResNet-50 | 1 |
| 02_resnet50_dual_site.py | ResNet-50 | 2 |
| 03_convnext_tiny_dual_site.py | ConvNeXt-Tiny | 2 |
| 04_efficientnet_b0_dual_site.py | EfficientNet-B0 | 2 |

Each script:
1. Loads and preprocesses data
2. Trains with GroupKFold validation (split by experiment)
3. Generates submission.csv

## Output

- `submission.csv` - predictions for test set
- `best_model.pth` - saved model weights
- `label_mapping.npy` - label encoder classes
