"""Data modules for v2 architecture."""

from .brats import BraTS2023DataModule
from .medmnist import MedMNIST3DDataModule

__all__ = ["BraTS2023DataModule", "MedMNIST3DDataModule"]
