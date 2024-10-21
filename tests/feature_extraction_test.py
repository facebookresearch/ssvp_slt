import unittest
from translation.feature_extraction_module import FeatureExtractionConfig, FeatureExtractionModule, LauncherConfig
from unittest.mock import patch, MagicMock

class TestFeatureExtractionModule(unittest.TestCase):
    def setUp(self):
        # Mock the configuration for the FeatureExtractionModule
        self.config = FeatureExtractionConfig(
            data_dir="dailymoth-70h/blurred_clips",
            pretrained_model_path="signhiera_mock.pth", 
            launcher=LauncherConfig(cluster="normal")
        )
        self.module = FeatureExtractionModule(self.config)

    @patch("torch.cuda.is_available", return_value=False)  # Mock CUDA for CPU
    def test_load_model(self):
        # Test if the model loads properly
        model = self.module.load_model()
        self.assertIsNotNone(model)

    @patch("ssvp_slt.data.video_dataset.VideoDataset")
    def test_get_dataloader(self, mock_video_dataset):
        # Mock the VideoDataset and test the dataloader
        mock_dataset = MagicMock()
        mock_video_dataset.return_value = mock_dataset
        dataloader = self.module.get_dataloader((0, 10))
        self.assertIsNotNone(dataloader)

    @patch("torch.save")
    @patch("ssvp_slt.util.misc.Prefetcher")
    @patch("torch.cuda.is_available", return_value=False)
    def test_run(self, mock_cuda, mock_prefetcher, mock_torch_save):
        # Mock dependencies inside the run method
        mock_batch = {"frames": MagicMock(), "padding": MagicMock()}
        mock_prefetcher.return_value = MagicMock()
        mock_prefetcher.return_value.__next__.side_effect = [mock_batch, None]

        # Run the feature extraction for shard 0
        self.module.run(iteration_value=1, iteration_index=0)

        # Assert that prefetcher is being iterated, and the process is triggered
        self.assertTrue(mock_prefetcher.called)

if __name__ == "__main__":
    unittest.main()
