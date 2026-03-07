from __future__ import annotations

from typing import Literal

from lightning import LightningDataModule
from monai.transforms.compose import Compose
from monai.transforms.utility.array import EnsureChannelFirst
from torch.utils.data import DataLoader

from maskgit3d.infrastructure.data import medmnist_provider
from maskgit3d.infrastructure.data.transforms import (
    create_medmnist_inference_preprocessing,
    create_medmnist_training_preprocessing,
)


class MedMNIST3DDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_type: str = "organ",
        crop_size: tuple[int, int, int] = (32, 32, 32),
        input_size: int = 28,
        batch_size: int = 16,
        num_workers: int = 4,
        data_root: str = "./data",
        download: bool = True,
        pin_memory: bool = True,
        drop_last_train: bool = True,
    ) -> None:
        super().__init__()

        try:
            self.dataset_type = medmnist_provider.MedMNIST3DDataset(dataset_type.lower())
        except ValueError as exc:
            supported = [item.value for item in medmnist_provider.MedMNIST3DDataset]
            raise ValueError(
                f"Unsupported dataset type: '{dataset_type}'. Supported types: {supported}"
            ) from exc

        if input_size not in medmnist_provider.MedMnist3DDataProvider.SUPPORTED_INPUT_SIZES:
            raise ValueError(
                "input_size must be one of "
                f"{medmnist_provider.MedMnist3DDataProvider.SUPPORTED_INPUT_SIZES}, got {input_size}"
            )

        self.crop_size = crop_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.download = download
        self.pin_memory = pin_memory
        self.drop_last_train = drop_last_train

        base_train_transform = create_medmnist_training_preprocessing(
            crop_size=self.crop_size,
            input_size=self.input_size,
        )
        base_inference_transform = create_medmnist_inference_preprocessing()

        self.train_transform: Compose = Compose(
            [EnsureChannelFirst(channel_dim="no_channel"), *base_train_transform.transforms]
        )
        self.inference_transform: Compose = Compose(
            [EnsureChannelFirst(channel_dim="no_channel"), *base_inference_transform.transforms]
        )

        self._train_dataset: medmnist_provider.MedMNIST3DDatasetWrapper | None = None
        self._val_dataset: medmnist_provider.MedMNIST3DDatasetWrapper | None = None
        self._test_dataset: medmnist_provider.MedMNIST3DDatasetWrapper | None = None

    def _create_dataset(
        self,
        split: Literal["train", "val", "test"],
        transform: Compose,
    ) -> medmnist_provider.MedMNIST3DDatasetWrapper:
        dataset_class = medmnist_provider._get_dataset_class(self.dataset_type)
        medmnist_split = "val" if split == "val" else split

        base_dataset = dataset_class(
            root=self.data_root,
            split=medmnist_split,
            download=self.download,
            size=self.input_size,
        )

        return medmnist_provider.MedMNIST3DDatasetWrapper(
            dataset=base_dataset,
            transform=transform,
            spatial_size=self.crop_size,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if self._train_dataset is None:
                self._train_dataset = self._create_dataset("train", transform=self.train_transform)
            if self._val_dataset is None:
                self._val_dataset = self._create_dataset("val", transform=self.inference_transform)

        if stage in (None, "test"):
            if self._test_dataset is None:
                self._test_dataset = self._create_dataset(
                    "test", transform=self.inference_transform
                )

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")

        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last_train,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader()")

        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")

        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
