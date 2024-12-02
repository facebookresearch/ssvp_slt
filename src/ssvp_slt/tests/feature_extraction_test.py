# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from ssvp_slt.translation.feature_extraction_module import FeatureExtractionConfig, FeatureExtractionModule, LauncherConfig
from unittest.mock import patch, MagicMock
from ssvp_slt.utils.download_model import download_model

class TestFeatureExtractionModule(unittest.TestCase):
    def setUp(self):
        # Mock the configuration for the FeatureExtractionModule
        model_path = download_model('https://dl.fbaipublicfiles.com/SONAR/asl/signhiera_mock.pth')
        self.config = FeatureExtractionConfig(
            data_dir="MOCK_dataset",
            pretrained_model_path=model_path, 
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
