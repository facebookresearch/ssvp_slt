# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import sys
import os
from omegaconf import OmegaConf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now, import the modules
from translation.run_translation_module import Config, run_translation, ModelConfig, DataConfig, CommonConfig


# Define the translation configuration
translation_config = Config(
    common=CommonConfig(
        eval=True,
        load_model="translation/signhiera_mock.pth",
        num_workers=0
    ),
    data=DataConfig(     
        val_data_dir="features_outputs/0"
    ),
    model=ModelConfig(
        name_or_path="google-t5/t5-base",
        feature_dim=768
    )
)

# Convert it to DictConfig
translation_dict_config = OmegaConf.structured(translation_config)

# Run translation with the DictConfig instance
run_translation(translation_dict_config)
