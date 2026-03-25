"""Tests for BraTS mixed-modality sampling (4-channel -> single-channel + label)."""

from pathlib import Path

import pytest
import torch

from maskgit3d.data.brats.config import BraTSSubDataset
from maskgit3d.data.brats.dataset import BraTS2023CaseRecord, BraTS2023Dataset


class TestMixedModalitySampling:
    """Test dataset returns single modality with label instead of 4-channel."""

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

    def test_getitem_returns_single_path_not_list(self) -> None:
        """Dataset __getitem__ should return single path, not list of 4."""
        cases = self.create_mock_cases(1)
        dataset = BraTS2023Dataset(cases=cases, transform=None)

        result = dataset[0]

        # NEW CONTRACT: "image" is a single Path, not list
        assert isinstance(result["image"], Path), f"Expected Path, got {type(result['image'])}"
        assert (
            "t1n" in str(result["image"])
            or "t1c" in str(result["image"])
            or "t2w" in str(result["image"])
            or "t2f" in str(result["image"])
        )

    def test_getitem_returns_modality_label(self) -> None:
        """Dataset __getitem__ should return numeric modality label."""
        cases = self.create_mock_cases(1)
        dataset = BraTS2023Dataset(cases=cases, transform=None)

        result = dataset[0]

        # NEW CONTRACT: "modality_label" is int in [0, 3]
        assert "modality_label" in result, "Missing modality_label key"
        assert isinstance(result["modality_label"], int)
        assert 0 <= result["modality_label"] <= 3

    def test_getitem_returns_modality_name(self) -> None:
        """Dataset __getitem__ should return modality name for reference."""
        cases = self.create_mock_cases(1)
        dataset = BraTS2023Dataset(cases=cases, transform=None)

        result = dataset[0]

        # NEW CONTRACT: "modality" is string name
        assert "modality" in result, "Missing modality key"
        assert isinstance(result["modality"], str)
        assert result["modality"] in ["t1n", "t1c", "t2w", "t2f"]

    def test_modality_label_mapping(self) -> None:
        """Test modality label mapping: t1n=0, t1c=1, t2w=2, t2f=3."""
        from maskgit3d.data.brats.dataset import MODALITY_TO_LABEL

        expected = {"t1n": 0, "t1c": 1, "t2w": 2, "t2f": 3}
        assert MODALITY_TO_LABEL == expected

    def test_same_case_can_return_different_modalities(self) -> None:
        """Multiple accesses to same case can return different modalities (stochastic)."""
        cases = self.create_mock_cases(1)
        dataset = BraTS2023Dataset(cases=cases, transform=None)

        # Sample multiple times and collect modalities
        modalities = set()
        for _ in range(20):
            result = dataset[0]
            modalities.add(result["modality"])

        # With enough samples, should see variety
        # (probabilistic, but highly likely with 20 samples from 4 options)
        assert len(modalities) > 1, "Expected different modalities across samples"

    def test_case_id_and_subdataset_preserved(self) -> None:
        """case_id and subdataset metadata should be preserved."""
        cases = self.create_mock_cases(1)
        dataset = BraTS2023Dataset(cases=cases, transform=None)

        result = dataset[0]

        assert result["case_id"] == "BraTS-GLI-00000-000"
        assert result["subdataset"] == BraTSSubDataset.GLI

    def test_getitem_with_transform_applies_to_sample(self) -> None:
        """Transform should receive single-modality dict sample."""
        from unittest.mock import MagicMock

        cases = self.create_mock_cases(1)
        mock_transform = MagicMock(return_value={"transformed": True})
        dataset = BraTS2023Dataset(cases=cases, transform=mock_transform)

        dataset[0]

        mock_transform.assert_called_once()
        call_args = mock_transform.call_args[0][0]
        # Transform receives single path dict
        assert isinstance(call_args["image"], Path)
        assert "modality_label" in call_args


class TestDeterministicValidationSampling:
    """Test validation sampling can be made deterministic."""

    def create_mock_cases(self, n: int = 2) -> list:
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

    def test_deterministic_sampling_with_same_seed(self) -> None:
        """Same seed should produce same modality selection."""
        cases = self.create_mock_cases(5)
        dataset = BraTS2023Dataset(cases=cases, transform=None, deterministic=True, seed=42)

        result1 = dataset[0]
        result2 = dataset[0]

        assert result1["modality_label"] == result2["modality_label"]
        assert result1["modality"] == result2["modality"]

    def test_different_seeds_produce_different_modalities(self) -> None:
        """Different seeds should potentially produce different selections."""
        cases = self.create_mock_cases(5)
        dataset1 = BraTS2023Dataset(cases=cases, transform=None, deterministic=True, seed=42)
        dataset2 = BraTS2023Dataset(cases=cases, transform=None, deterministic=True, seed=123)

        mods1 = [dataset1[i]["modality"] for i in range(len(cases))]
        mods2 = [dataset2[i]["modality"] for i in range(len(cases))]

        assert mods1 != mods2

    def test_deterministic_is_case_based_not_index_based(self) -> None:
        """Same case should get same modality regardless of position in list."""
        cases = self.create_mock_cases(3)

        dataset_normal = BraTS2023Dataset(cases=cases, transform=None, deterministic=True, seed=42)
        modality_first_pos = dataset_normal[0]["modality"]

        reversed_cases = list(reversed(cases))
        dataset_reversed = BraTS2023Dataset(
            cases=reversed_cases, transform=None, deterministic=True, seed=42
        )
        original_first_case = cases[0]
        modality_in_reversed = None
        for i, case in enumerate(reversed_cases):
            if case.case_id == original_first_case.case_id:
                modality_in_reversed = dataset_reversed[i]["modality"]
                break

        assert modality_in_reversed is not None
        assert modality_first_pos == modality_in_reversed


class TestModalityLabelMapping:
    """Test fixed modality label mapping."""

    def test_modality_order_mapping(self) -> None:
        """Test MODALITY_ORDER maps to labels 0-3."""
        from maskgit3d.data.brats.config import MODALITY_ORDER

        expected_mapping = {
            "t1n": 0,
            "t1c": 1,
            "t2w": 2,
            "t2f": 3,
        }

        for label, modality in enumerate(MODALITY_ORDER):
            assert expected_mapping[modality] == label
