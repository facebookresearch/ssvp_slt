# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os
import time
from pathlib import Path
from typing import Sequence
import torch
from omegaconf import DictConfig, OmegaConf
from configs import FeatureExtractionConfig, RunConfig, TranslationConfig
#TODO: add Preprocessor class
from feature_extraction import FeatureExtractor
from translation import Translator

def print_translations(keys: Sequence[str], translations: Sequence[str]) -> None:
    assert len(keys) == len(translations)
    print(f"\nTranslations:\n{'-'*50}")
    for key, translation in zip(keys, translations):
        print(f'{key}: "{translation}"')
    print(f"{'-' * 50}\n")
    
def run_pipeline(config: DictConfig):
    """
    Main function to run the sign language translation pipeline.
    """
    os.chdir(os.getcwd())  # Ensure correct working directory
    
    if config.verbose:
        print(f"Config:\n{OmegaConf.to_yaml(config, resolve=True)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if config.feature_extraction.fp16 else torch.float32

    t0 = time.time()

    # Initialize components
    feature_extractor = FeatureExtractor(config.feature_extraction, device=device)
    translator_cls = Translator
    translator = translator_cls(config.translation, device=device, dtype=dtype)
    
    t1 = time.time()
    if config.verbose:
        print(f"1. Model loading: {t1 - t0:.3f}s")
        
    #TODO: perform preprocessing
    extracted_features = feature_extractor(**inputs)    

    translations = translator(extracted_features)["translations"]

    keys = range(config.translation.num_translations)

    # Output results
    print_translations(keys, translations)


# Define the translation configuration
translation_config = RunConfig(
    video_path="path/to/your/video.mp4",
    verbose=True,
    feature_extraction=FeatureExtractionConfig(
        pretrained_model_path="path/to/your/model.pth",     
    ),
    translation=TranslationConfig(
        base_model_name="google-t5/t5-base",      
    )
)

# Convert it to DictConfig
translation_dict_config = OmegaConf.structured(translation_config)

# Run pipeline with provided parameters
run_pipeline(translation_dict_config)
