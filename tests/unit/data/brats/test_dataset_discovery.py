"""Tests for BraTS2023 case discovery and split logic."""

from pathlib import Path

from maskgit3d.data.brats.config import BraTSSubDataset
from maskgit3d.data.brats.dataset import (
    BraTS2023CaseRecord,
    _discover_cases,
    _generate_stratified_split,
    _is_complete_case,
)


class TestBraTS2023CaseRecord:
    """Test case record dataclass."""

    def test_case_record_creation(self) -> None:
        """Test creating a case record."""
        record = BraTS2023CaseRecord(
            case_id="BraTS-GLI-00001-000",
            subdataset=BraTSSubDataset.GLI,
            image_paths=[
                Path("/data/BraTS-GLI-00001-000/BraTS-GLI-00001-000-t1n.nii.gz"),
                Path("/data/BraTS-GLI-00001-000/BraTS-GLI-00001-000-t1c.nii.gz"),
                Path("/data/BraTS-GLI-00001-000/BraTS-GLI-00001-000-t2w.nii.gz"),
                Path("/data/BraTS-GLI-00001-000/BraTS-GLI-00001-000-t2f.nii.gz"),
            ],
        )
        assert record.case_id == "BraTS-GLI-00001-000"
        assert record.subdataset == BraTSSubDataset.GLI
        assert len(record.image_paths) == 4


class TestIsCompleteCase:
    """Test case completeness checking."""

    def test_complete_case_all_modalities_present(self, tmp_path: Path) -> None:
        """Test that case with all 4 modalities is complete."""
        case_dir = tmp_path / "BraTS-GLI-00001-000"
        case_dir.mkdir()
        for mod in ["t1n", "t1c", "t2w", "t2f"]:
            (case_dir / f"BraTS-GLI-00001-000-{mod}.nii.gz").touch()

        result = _is_complete_case(case_dir, "BraTS-GLI-00001-000")
        assert result is True

    def test_incomplete_case_missing_modality(self, tmp_path: Path) -> None:
        """Test that case missing a modality is incomplete."""
        case_dir = tmp_path / "BraTS-GLI-00001-000"
        case_dir.mkdir()
        for mod in ["t1n", "t1c", "t2w"]:  # Missing t2f
            (case_dir / f"BraTS-GLI-00001-000-{mod}.nii.gz").touch()

        result = _is_complete_case(case_dir, "BraTS-GLI-00001-000")
        assert result is False

    def test_incomplete_case_empty_directory(self, tmp_path: Path) -> None:
        """Test that empty case directory is incomplete."""
        case_dir = tmp_path / "BraTS-GLI-00001-000"
        case_dir.mkdir()

        result = _is_complete_case(case_dir, "BraTS-GLI-00001-000")
        assert result is False


class TestDiscoverCases:
    """Test case discovery from filesystem."""

    def test_discover_cases_all_subdatasets(self, tmp_path: Path) -> None:
        """Test discovering cases from all subdatasets."""
        # Create GLI cases
        for i in range(3):
            case_dir = tmp_path / f"BraTS-GLI-0000{i:02d}-000"
            case_dir.mkdir()
            for mod in ["t1n", "t1c", "t2w", "t2f"]:
                (case_dir / f"BraTS-GLI-0000{i:02d}-000-{mod}.nii.gz").touch()

        # Create MEN cases
        for i in range(2):
            case_dir = tmp_path / f"BraTS-MEN-0000{i:02d}-000"
            case_dir.mkdir()
            for mod in ["t1n", "t1c", "t2w", "t2f"]:
                (case_dir / f"BraTS-MEN-0000{i:02d}-000-{mod}.nii.gz").touch()

        cases = _discover_cases(tmp_path, [BraTSSubDataset.GLI, BraTSSubDataset.MEN])

        assert len(cases) == 5
        gli_cases = [c for c in cases if c.subdataset == BraTSSubDataset.GLI]
        men_cases = [c for c in cases if c.subdataset == BraTSSubDataset.MEN]
        assert len(gli_cases) == 3
        assert len(men_cases) == 2

    def test_discover_cases_skips_incomplete(self, tmp_path: Path) -> None:
        """Test that incomplete cases are skipped."""
        # Create complete case
        complete_dir = tmp_path / "BraTS-GLI-00001-000"
        complete_dir.mkdir()
        for mod in ["t1n", "t1c", "t2w", "t2f"]:
            (complete_dir / f"BraTS-GLI-00001-000-{mod}.nii.gz").touch()

        # Create incomplete case
        incomplete_dir = tmp_path / "BraTS-GLI-00002-000"
        incomplete_dir.mkdir()
        for mod in ["t1n", "t1c"]:  # Missing t2w, t2f
            (incomplete_dir / f"BraTS-GLI-00002-000-{mod}.nii.gz").touch()

        cases = _discover_cases(tmp_path, [BraTSSubDataset.GLI])

        assert len(cases) == 1
        assert cases[0].case_id == "BraTS-GLI-00001-000"

    def test_discover_cases_modality_order_fixed(self, tmp_path: Path) -> None:
        """Test that discovered cases have modalities in fixed order."""
        case_dir = tmp_path / "BraTS-GLI-00001-000"
        case_dir.mkdir()
        # Create modalities in random order
        for mod in ["t2f", "t1n", "t2w", "t1c"]:
            (case_dir / f"BraTS-GLI-00001-000-{mod}.nii.gz").touch()

        cases = _discover_cases(tmp_path, [BraTSSubDataset.GLI])

        assert len(cases) == 1
        # Check modalities are in fixed order: t1n, t1c, t2w, t2f
        paths = cases[0].image_paths
        assert "t1n" in str(paths[0])
        assert "t1c" in str(paths[1])
        assert "t2w" in str(paths[2])
        assert "t2f" in str(paths[3])

    def test_discover_cases_ignores_non_brats_directories(self, tmp_path: Path) -> None:
        """Test that non-BraTS directories are ignored."""
        # Create BraTS case
        brats_dir = tmp_path / "BraTS-GLI-00001-000"
        brats_dir.mkdir()
        for mod in ["t1n", "t1c", "t2w", "t2f"]:
            (brats_dir / f"BraTS-GLI-00001-000-{mod}.nii.gz").touch()

        # Create non-BraTS directory
        other_dir = tmp_path / "some_other_dir"
        other_dir.mkdir()
        (other_dir / "file.nii.gz").touch()

        cases = _discover_cases(tmp_path, [BraTSSubDataset.GLI])

        assert len(cases) == 1
        assert cases[0].case_id == "BraTS-GLI-00001-000"


class TestGenerateStratifiedSplit:
    """Test stratified split generation."""

    def create_mock_cases(self, n_gli: int, n_men: int, n_met: int) -> list:
        """Helper to create mock cases."""
        cases = []
        for i in range(n_gli):
            cases.append(
                BraTS2023CaseRecord(
                    case_id=f"GLI-{i:03d}",
                    subdataset=BraTSSubDataset.GLI,
                    image_paths=[
                        Path(f"/data/{i}-{mod}.nii.gz") for mod in ["t1n", "t1c", "t2w", "t2f"]
                    ],
                )
            )
        for i in range(n_men):
            cases.append(
                BraTS2023CaseRecord(
                    case_id=f"MEN-{i:03d}",
                    subdataset=BraTSSubDataset.MEN,
                    image_paths=[
                        Path(f"/data/{i}-{mod}.nii.gz") for mod in ["t1n", "t1c", "t2w", "t2f"]
                    ],
                )
            )
        for i in range(n_met):
            cases.append(
                BraTS2023CaseRecord(
                    case_id=f"MET-{i:03d}",
                    subdataset=BraTSSubDataset.MET,
                    image_paths=[
                        Path(f"/data/{i}-{mod}.nii.gz") for mod in ["t1n", "t1c", "t2w", "t2f"]
                    ],
                )
            )
        return cases

    def test_split_respects_train_ratio(self) -> None:
        """Test that split respects train_ratio."""
        cases = self.create_mock_cases(10, 10, 10)
        train_cases, held_out_cases = _generate_stratified_split(cases, train_ratio=0.8, seed=42)

        assert len(train_cases) == 24  # 80% of 30
        assert len(held_out_cases) == 6  # 20% of 30

    def test_split_is_stratified(self) -> None:
        """Test that split preserves subdataset proportions."""
        cases = self.create_mock_cases(10, 10, 10)
        train_cases, held_out_cases = _generate_stratified_split(cases, train_ratio=0.8, seed=42)

        # Check stratification
        train_gli = [c for c in train_cases if c.subdataset == BraTSSubDataset.GLI]
        train_men = [c for c in train_cases if c.subdataset == BraTSSubDataset.MEN]
        train_met = [c for c in train_cases if c.subdataset == BraTSSubDataset.MET]

        held_gli = [c for c in held_out_cases if c.subdataset == BraTSSubDataset.GLI]
        held_men = [c for c in held_out_cases if c.subdataset == BraTSSubDataset.MEN]
        held_met = [c for c in held_out_cases if c.subdataset == BraTSSubDataset.MET]

        # Each subdataset should have ~80% in train
        assert len(train_gli) == 8
        assert len(train_men) == 8
        assert len(train_met) == 8

        assert len(held_gli) == 2
        assert len(held_men) == 2
        assert len(held_met) == 2

    def test_split_is_deterministic_with_same_seed(self) -> None:
        """Test that same seed produces same split."""
        cases = self.create_mock_cases(10, 10, 10)

        train1, held1 = _generate_stratified_split(cases, train_ratio=0.8, seed=42)
        train2, held2 = _generate_stratified_split(cases, train_ratio=0.8, seed=42)

        assert [c.case_id for c in train1] == [c.case_id for c in train2]
        assert [c.case_id for c in held1] == [c.case_id for c in held2]

    def test_split_is_different_with_different_seed(self) -> None:
        """Test that different seeds produce different splits."""
        cases = self.create_mock_cases(10, 10, 10)

        train1, held1 = _generate_stratified_split(cases, train_ratio=0.8, seed=42)
        train2, held2 = _generate_stratified_split(cases, train_ratio=0.8, seed=123)

        assert [c.case_id for c in train1] != [c.case_id for c in train2]

    def test_all_cases_included_in_split(self) -> None:
        """Test that all cases are in either train or held-out."""
        cases = self.create_mock_cases(10, 10, 10)
        train_cases, held_out_cases = _generate_stratified_split(cases, train_ratio=0.8, seed=42)

        all_ids = {c.case_id for c in train_cases + held_out_cases}
        original_ids = {c.case_id for c in cases}

        assert all_ids == original_ids
        assert len(all_ids) == 30

    def test_train_and_held_out_are_disjoint(self) -> None:
        """Test that train and held-out sets don't overlap."""
        cases = self.create_mock_cases(10, 10, 10)
        train_cases, held_out_cases = _generate_stratified_split(cases, train_ratio=0.8, seed=42)

        train_ids = {c.case_id for c in train_cases}
        held_ids = {c.case_id for c in held_out_cases}

        assert len(train_ids.intersection(held_ids)) == 0
