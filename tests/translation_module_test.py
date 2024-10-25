import unittest
from unittest.mock import MagicMock, patch
from translation.run_translation_module import TranslationModule, Config
import numpy as np

class TestTranslationModule(unittest.TestCase):
    def setUp(self):
        # Basic setup for testing TranslationModule
        self.config = Config()
        self.config.common.load_model = "mock_model_path"
        self.config.model.name_or_path = "mock_model_path"
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
        self.assertEqual(self.config.common.load_model, "mock_model_path")
        self.assertEqual(self.config.model.name_or_path, "mock_model_path")

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
