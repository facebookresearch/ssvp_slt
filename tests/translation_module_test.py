# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import unittest
from unittest.mock import MagicMock, patch
from translation.run_translation_module import TranslationModule, Config
import numpy as np

class TestTranslationModule(unittest.TestCase):
    def setUp(self):
        # Basic setup for testing TranslationModule
        self.config = Config()
        self.config.model.name_or_path = "translation/signhiera_mock.pth"
        self.translator = TranslationModule(self.config)

    @patch("run_translation_module.TranslationModule.run_translation")
    def test_translation_with_mock_features(self, mock_run_translation):
        # Mock feature array that simulates extracted features
        mock_features = np.random.rand(10, 512)  # 10 timesteps, 512-dim features

        # Mock translation return value
        mock_run_translation.return_value = "This is a test translation."

        # Run translation with mocked features
        result = self.translator.run_translation(mock_features)

        # Assertions
        self.assertEqual(result, "This is a test translation.")
        self.assertTrue(mock_run_translation.called)
        mock_run_translation.assert_called_with(mock_features)

    def test_configuration_loading(self):
        # Ensure the configuration fields are loaded as expected
        self.assertEqual(self.config.model.name_or_path, "translation/signhiera_mock.pth")

    @patch("translation_module.TranslationModule.run_translation")
    def test_translation_output_type(self, mock_run_translation):
        # Mock feature array for translation
        mock_features = np.random.rand(10, 512)
        
        # Mock output for translation to simulate text output
        mock_run_translation.return_value = "Translation successful."

        # Perform translation
        output = self.translator.run_translation(mock_features)

        # Assertions
        self.assertIsInstance(output, str)  # Check output type
        self.assertTrue(mock_run_translation.called)
