# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from ssvp_slt.modeling.sign_hiera import FeatureExtractor
from ssvp_slt.modeling.sonar import SonarTranslator
from ssvp_slt.util.video import Preprocessor


def print_translations(keys: Sequence[str], translations: Sequence[str]) -> None:
    assert len(keys) == len(translations)
    print(f"\nTranslations:\n{'-'*50}")
    for key, translation in zip(keys, translations):
        print(f'{key}: "{translation}"')
    print(f"{'-' * 50}\n")


@dataclass
class PreprocessingConfig:
    # Bounding box expansion and threshold parameters
    up_exp: float = 1.0
    down_exp: float = 3.0
    left_exp: float = 1.5
    right_exp: float = 1.5
    iou_threshold: float = 0.2
    num_ratio_threshold: float = 0.5

    # Dlib detection parameters
    hog_detector: bool = False
    detector_path: Optional[str] = MISSING
    detection_sampling_rate: int = 16
    detection_downsample: bool = False

    # Sliding window sampling parameters
    num_frames: int = 128
    feature_extraction_stride: int = 64
    sampling_rate: int = 2
    target_fps: int = 25

    # Cropping and Normalization
    target_size: int = 224
    mean: Tuple[float, float, float] = (0.45, 0.45, 0.45)
    std: Tuple[float, float, float] = (0.225, 0.225, 0.225)

    debug: bool = False
    verbose: bool = False

    def __post_init__(self):
        if self.hog_detector is False:
            assert (
                self.detector_path is not None
            ), "detector_path must be provded if `hog_detector=False`"


@dataclass
class FeatureExtractionConfig:
    pretrained_model_path: str = MISSING
    model_name: str = "hiera_base_128x224"
    max_batch_size: int = 2
    fp16: bool = True
    verbose: bool = False


@dataclass
class TranslationConfig:
    pretrained_model_path: str = MISSING
    base_model_name: str = "google/t5-v1_1-large"
    tgt_langs: List[str] = field(default_factory=lambda: ["eng_Latn"])
    verbose: bool = False


@dataclass
class RunConfig:
    video_path: str = MISSING
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    feature_extraction: FeatureExtractionConfig = FeatureExtractionConfig()
    translation: TranslationConfig = TranslationConfig()
    verbose: bool = False


cs = ConfigStore.instance()
cs.store(name="run_config", node=RunConfig)


@hydra.main(config_name="run_config")
def main(config: DictConfig):

    os.chdir(hydra.utils.get_original_cwd())

    if config.verbose:
        print(f"Config:\n{OmegaConf.to_yaml(config, resolve=True)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if config.feature_extraction.fp16 else torch.float32

    t0 = time.time()

    preprocessor = Preprocessor(config.preprocessing, device=device)
    feature_extractor = FeatureExtractor(config.feature_extraction, device=device)

    translator_cls = SonarTranslator
    translator = translator_cls(config.translation, device=device, dtype=dtype)

    t1 = time.time()

    if config.verbose:
        print(f"Model loading: {t1 - t0:.3f}s")

    inputs = preprocessor(Path(config.video_path))
    extracted_features = feature_extractor(**inputs)

    kwargs = {"tgt_langs": config.translation.tgt_langs}
    translations = translator(extracted_features, **kwargs)["translations"]

    print_translations(config.translation.tgt_langs, translations)


if __name__ == "__main__":
    main()
