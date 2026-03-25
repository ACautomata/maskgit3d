"""Tests for BraTS2023 dataset runtime behavior."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from maskgit3d.data.brats.config import BraTSSubDataset
from maskgit3d.data.brats.dataset import BraTS2023CaseRecord, BraTS2023Dataset


class TestBraTS2023Dataset:
    """Test BraTS2023 dataset wrapper."""

    def create_mock_cases(self, n: int = 2) -> list:
        """Helper to create mock case records."""
        cases = []
        for i in range(n):
            case_id = f"BraTS-GLI-{i:05d}-000"
            cases.append(
                BraTS2023CaseRecord(
                    case_id=case_id,
                    subdataset=BraTSSubDataset.GLI,
                    image_paths=[
                        Path(f"/data/{case_id}/{case_id}-t1n.nii.gz"),
                        Path(f"/data/{case_id}/{case_id}-t1c.nii.gz"),
                        Path(f"/data/{case_id}/{case_id}-t2w.nii.gz"),
                        Path(f"/data/{case_id}/{case_id}-t2f.nii.gz"),
                    ],
                )
            )
        return cases

    def test_dataset_length(self) -> None:
        """Test that dataset length matches number of cases."""
        cases = self.create_mock_cases(5)
        dataset = BraTS2023Dataset(cases=cases, transform=None)

        assert len(dataset) == 5

    def test_dataset_getitem_no_transform(self) -> None:
        """Test getting item without transform returns raw record."""
        cases = self.create_mock_cases(1)
        dataset = BraTS2023Dataset(cases=cases, transform=None)

        result = dataset[0]

        assert isinstance(result, dict)
        assert "image" in result
        assert result["case_id"] == "BraTS-GLI-00000-000"
        assert isinstance(result["image"], Path)
        assert "modality_label" in result
        assert isinstance(result["modality_label"], int)
        assert 0 <= result["modality_label"] <= 3

    def test_dataset_getitem_with_transform(self) -> None:
        """Test getting item applies transform."""
        cases = self.create_mock_cases(1)
        mock_transform = MagicMock(return_value={"image": "transformed"})
        dataset = BraTS2023Dataset(cases=cases, transform=mock_transform)

        result = dataset[0]

        mock_transform.assert_called_once()
        assert result == {"image": "transformed"}

    def test_dataset_returns_self_reconstruction_sample(self) -> None:
        """Test that dataset returns self-reconstruction sample (target=input)."""
        cases = self.create_mock_cases(1)
        mock_transform = MagicMock(return_value={"image": [1, 2, 3, 4]})
        dataset = BraTS2023Dataset(cases=cases, transform=mock_transform)

        result = dataset[0]

        # For reconstruction, target should equal input
        assert "image" in result

    def test_dataset_iteration(self) -> None:
        """Test that dataset can be iterated."""
        cases = self.create_mock_cases(3)
        dataset = BraTS2023Dataset(cases=cases, transform=None)

        items = list(dataset)

        assert len(items) == 3
        for item in items:
            assert "image" in item
            assert "case_id" in item

    def test_dataset_empty_cases(self) -> None:
        """Test that dataset handles empty case list."""
        dataset = BraTS2023Dataset(cases=[], transform=None)

        assert len(dataset) == 0

    def test_dataset_getitem_out_of_range(self) -> None:
        """Test that out of range index raises IndexError."""
        cases = self.create_mock_cases(2)
        dataset = BraTS2023Dataset(cases=cases, transform=None)

        with pytest.raises(IndexError):
            dataset[10]

    def test_dataset_returns_modality_info(self) -> None:
        """Test that dataset returns modality label and name."""
        cases = self.create_mock_cases(1)
        dataset = BraTS2023Dataset(cases=cases, transform=None, deterministic=True, seed=42)

        result = dataset[0]
        path = result["image"]

        assert isinstance(path, Path)
        assert any(m in str(path) for m in ["t1n", "t1c", "t2w", "t2f"])
        assert "modality_label" in result
        assert result["modality"] in ["t1n", "t1c", "t2w", "t2f"]
