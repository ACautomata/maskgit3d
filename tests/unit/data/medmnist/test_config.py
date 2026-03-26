"""Tests for MedMNIST configuration."""

from maskgit3d.data.medmnist.config import (
    MedMNISTConfig,
    MedMNISTDatasetName,
    TaskType,
)


class TestMedMNISTDatasetName:
    def test_enum_values(self):
        assert MedMNISTDatasetName.ORGAN.value == "organmnist3d"
        assert MedMNISTDatasetName.NODULE.value == "nodulemnist3d"
        assert MedMNISTDatasetName.ADRENAL.value == "adrenalmnist3d"
        assert MedMNISTDatasetName.VESSEL.value == "vesselmnist3d"
        assert MedMNISTDatasetName.FRACTURE.value == "fracturemnist3d"
        assert MedMNISTDatasetName.SYNAPSE.value == "synapsemnist3d"


class TestTaskType:
    def test_enum_values(self):
        assert TaskType.RECONSTRUCTION.value == "reconstruction"
        assert TaskType.CLASSIFICATION.value == "classification"


class TestMedMNISTConfig:
    def test_default_values(self):
        config = MedMNISTConfig(dataset_name=MedMNISTDatasetName.ORGAN)
        assert config.task_type == TaskType.RECONSTRUCTION
        assert config.crop_size == (64, 64, 64)
        assert config.batch_size == 32
        assert config.num_workers == 8
        assert config.pin_memory is True
        assert config.drop_last_train is True
        assert config.download is True

    def test_num_classes_per_dataset(self):
        test_cases = [
            (MedMNISTDatasetName.ORGAN, 11),
            (MedMNISTDatasetName.NODULE, 2),
            (MedMNISTDatasetName.ADRENAL, 2),
            (MedMNISTDatasetName.VESSEL, 2),
            (MedMNISTDatasetName.FRACTURE, 3),
            (MedMNISTDatasetName.SYNAPSE, 1),
        ]
        for dataset_name, expected in test_cases:
            config = MedMNISTConfig(dataset_name=dataset_name)
            assert config.num_classes == expected

    def test_input_size_property(self):
        config = MedMNISTConfig(dataset_name=MedMNISTDatasetName.ORGAN)
        assert config.input_size == 64

    def test_custom_values(self):
        config = MedMNISTConfig(
            dataset_name=MedMNISTDatasetName.NODULE,
            task_type=TaskType.CLASSIFICATION,
            crop_size=(64, 64, 64),
            batch_size=16,
        )
        assert config.dataset_name == MedMNISTDatasetName.NODULE
        assert config.task_type == TaskType.CLASSIFICATION
        assert config.crop_size == (64, 64, 64)
        assert config.batch_size == 16
        assert config.num_classes == 2
