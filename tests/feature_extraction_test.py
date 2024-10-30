import unittest
from translation.feature_extraction_module import FeatureExtractionConfig, FeatureExtractionModule, LauncherConfig
from unittest.mock import patch, MagicMock

class TestFeatureExtractionModule(unittest.TestCase):
    def setUp(self):
        # Mock the configuration for the FeatureExtractionModule
        self.config = FeatureExtractionConfig(
            data_dir="MOCK_dataset",
            pretrained_model_path="signhiera_mock.pth", 
            launcher=LauncherConfig(cluster="local")
        )
        self.module = FeatureExtractionModule(self.config)

    @patch("torch.cuda.is_available", return_value=False)  # Mock CUDA for CPU
    def test_load_model(self, mock_cuda):
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

if __name__ == "__main__":
    unittest.main()
