# Wavelet-Based Low-Frequency Enhancement for Ultrasound Medical Image Segmentation

This repository accompanies the paper **"Wavelet-Based Low-Frequency Enhancement for Ultrasound Medical Image Segmentation."**

## Datasets

The experiments use four publicly available ultrasound datasets:

| Dataset | Task | Dataset paper |
| --- | --- | --- |
| DDTI | Thyroid nodule segmentation | [Pedraza et al., 2015](https://doi.org/10.1117/12.2073532) |
| TN3K | Thyroid nodule segmentation | [Gong et al., 2021](https://doi.org/10.1109/ISBI48211.2021.9434087) |
| BUSI | Breast lesion segmentation | [Al-Dhabyani et al., 2020](https://doi.org/10.1016/j.dib.2019.104863) |
| BUS-BRA | Breast lesion segmentation | [Gomez-Flores et al., 2024](https://doi.org/10.1002/mp.16812) |

The train/test and five-fold split files are provided in [`datasets/`](datasets/).

## Repository structure

- [`models/WLENet.py`](models/WLENet.py): revised WLENet with a reconstructed db2 reference and discrepancy- and gradient-conditioned gating.
- [`models/legacy/WLENet.py`](models/legacy/WLENet.py): earlier WLENet implementation with direct Haar LL guidance.
- [`datasets/`](datasets/): data loader and split definitions for five-fold cross-validation.
- [`evaluation/eval_HD95.py`](evaluation/eval_HD95.py): HD95 evaluation in resized-image pixels. Use a test loader with `batch_size=1` to average per-image HD95 values over the test set.
