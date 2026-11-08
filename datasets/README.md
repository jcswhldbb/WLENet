# Dataset Split Files

This directory contains the data loader and split definitions used in the experiments.

## Contents

- `DDTI/`, `TN3K/`, `BUSI/`, and `BUSBRA/`: five-fold split files.

Each split uses a fixed test set and five folds on the remaining training data (seed 42).

The BUSI split uses the cleaned lesion subset; multiple masks for one image are merged.

## Loader

`dataset.py` provides PyTorch dataset loaders. Place each dataset under
`data/<dataset>/`, for example:

```python
from datasets.dataset import BUSI

train_set = BUSI(mode="train", fold=0)
test_set = BUSI(mode="test")
```

Use `data_root` to specify another local data directory.
