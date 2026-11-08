# Dataset Split Files

This directory contains the split definitions used in the experiments. The source ultrasound images and masks are not included; obtain them from the dataset providers listed in the repository root README.

## Contents

- `DDTI/`: five train-validation fold JSON files.
- `TN3K/`: five train-validation fold JSON files, the complete train-validation JSON, and the original train/test label CSV files.
- `BUSI/`: five train-validation fold JSON files for the cleaned lesion-bearing BUSI subset.
- `BUSBRA/`: five train-validation fold JSON files.

For each dataset, the fixed test split is excluded from the five-fold construction. Each JSON file defines one train-validation fold; four folds are used for training and the remaining fold is used for validation. The random seed is 42.

The BUSI cleaning procedure was applied before splitting. It excludes normal images for lesion segmentation, removes audited duplicate and invalid cases, and merges multiple lesion masks for an image by pixel-wise union.
