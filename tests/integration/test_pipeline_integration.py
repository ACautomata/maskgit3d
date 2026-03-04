"""Integration tests for training pipeline components."""


from maskgit3d.infrastructure.data.dataset import SimpleDataProvider


class TestDataProviderIntegration:
    """Test data providers with real data flow."""

    def test_simple_data_provider_tensors(self):
        """Test SimpleDataProvider returns correct tensor shapes."""
        provider = SimpleDataProvider(
            num_train=4,
            num_val=2,
            num_test=2,
            batch_size=2,
            in_channels=1,
            out_channels=1,
            spatial_size=(32, 32, 32),
        )

        # Test train loader (returns DataLoader)
        batch_x, batch_y = next(iter(provider.train_loader()))

        assert batch_x.shape == (2, 1, 32, 32, 32)
        assert batch_y.shape == (2, 1, 32, 32, 32)

    def test_simple_data_provider_val_test_loaders(self):
        """Test validation and test loaders."""
        provider = SimpleDataProvider(
            num_train=4,
            num_val=2,
            num_test=2,
            batch_size=2,
            spatial_size=(16, 16, 16),
        )

        val_x, val_y = next(iter(provider.val_loader()))
        test_x, test_y = next(iter(provider.test_loader()))

        assert val_x.shape[0] <= 2
        assert test_x.shape[0] <= 2
