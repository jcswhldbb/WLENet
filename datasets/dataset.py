"""Dataset loaders for the published WLENet train/validation splits.

The repository distributes split definitions only. Download the original images
and masks separately, then arrange each dataset as::

    <data_root>/<dataset>/
        trainval-image/
        trainval-mask/
        test-image/
        test-mask/

For example, ``DDTI(mode="train", fold=0)`` uses the split file in
``datasets/DDTI/`` and images in ``./data/DDTI/`` by default.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SPLIT_ROOT = Path(__file__).resolve().parent


class UltrasoundSegmentationDataset(data.Dataset):
    """Base class for a dataset with fixed test data and five CV folds."""

    dataset_name: str
    split_prefix: str
    zero_based_split_indices = False

    def __init__(
        self,
        mode: str,
        transform: Optional[Callable] = None,
        fold: int = 0,
        data_root: Optional[Union[str, Path]] = None,
        mask_size: int = 224,
    ) -> None:
        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be one of: 'train', 'val', or 'test'.")
        if not 0 <= fold < 5:
            raise ValueError("fold must be an integer from 0 to 4.")

        self.mode = mode
        self.transform = transform
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (mask_size, mask_size),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

        self.data_dir = self._resolve_data_dir(data_root)
        self.images = self._build_samples(fold)

    def _resolve_data_dir(self, data_root: Optional[Union[str, Path]]) -> Path:
        if data_root is None:
            return _REPOSITORY_ROOT / "data" / self.dataset_name

        root = Path(data_root)
        dataset_dir = root / self.dataset_name
        return dataset_dir if dataset_dir.exists() else root

    def _split_file(self, fold: int) -> Path:
        return _SPLIT_ROOT / self.dataset_name / f"{self.split_prefix}-trainval-fold{fold}.json"

    @staticmethod
    def _numeric_file_list(directory: Path) -> List[Path]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing directory: {directory}")

        try:
            return sorted(directory.iterdir(), key=lambda path: int(path.stem))
        except ValueError as error:
            raise ValueError(
                f"All image filenames in {directory} must use a numeric stem, e.g. 1.png."
            ) from error

    def _samples_from_indices(self, indices: List[int]) -> List[Tuple[Path, Path]]:
        image_dir = self.data_dir / "trainval-image"
        mask_dir = self.data_dir / "trainval-mask"
        image_files = self._numeric_file_list(image_dir)

        samples = []
        for index in indices:
            position = index if self.zero_based_split_indices else index - 1
            if position < 0 or position >= len(image_files):
                raise IndexError(
                    f"Split index {index} is outside the available files in {image_dir}."
                )

            image_path = image_files[position]
            mask_path = mask_dir / image_path.name
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            samples.append((image_path, mask_path))
        return samples

    def _test_samples(self) -> List[Tuple[Path, Path]]:
        image_dir = self.data_dir / "test-image"
        mask_dir = self.data_dir / "test-mask"
        samples = []
        for image_path in self._numeric_file_list(image_dir):
            mask_path = mask_dir / image_path.name
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            samples.append((image_path, mask_path))
        return samples

    def _build_samples(self, fold: int) -> List[Tuple[Path, Path]]:
        if self.mode == "test":
            return self._test_samples()

        split_path = self._split_file(fold)
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        with split_path.open(encoding="utf-8") as handle:
            split = json.load(handle)
        return self._samples_from_indices(split[self.mode])

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_path, mask_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = mask.point(lambda value: 255 if value > 127 else 0)

        if self.transform is not None:
            image = self.transform(image)
        mask = self.mask_transform(mask)

        return {
            "image": image,
            "label": mask,
            "label_name": mask_path.name,
        }

    def __len__(self) -> int:
        return len(self.images)


class TN3K(UltrasoundSegmentationDataset):
    dataset_name = "TN3K"
    split_prefix = "tn3k"
    zero_based_split_indices = True


class DDTI(UltrasoundSegmentationDataset):
    dataset_name = "DDTI"
    split_prefix = "DDTI"


class BUSBRA(UltrasoundSegmentationDataset):
    dataset_name = "BUSBRA"
    split_prefix = "BUSBRA"


class BUSI(UltrasoundSegmentationDataset):
    dataset_name = "BUSI"
    split_prefix = "BUSI"
