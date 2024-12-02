
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from omegaconf import II, MISSING


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
    hog_detector: bool = True
    detector_path: Optional[str] = "checkpoints/detector.dat"
    detection_sampling_rate: int = 16
    detection_downsample: bool = True

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
    verbose: bool = II("verbose")

    def __post_init__(self):
        if self.hog_detector is False:
            assert (
                self.detector_path is not None
            ), "detector_path must be provded if `hog_detector=False`"


@dataclass
class FeatureExtractionConfig:
    pretrained_model_path: str = "checkpoints/feature_extractor.pth"
    model_name: str = "hiera_base_128x224"
    max_batch_size: int = 2
    fp16: bool = True
    verbose: bool = II("verbose")


@dataclass
class TranslationConfig:
    pretrained_model_path: str = "checkpoints/translator.pth"
    tokenizer_path: str = "google-t5/t5-base"
    base_model_name: str = "google-t5/t5-base"
    feature_dim: int = 768
    decoder_path: str = "checkpoints/sonar_decoder.pt"
    decoder_spm_path: str = "checkpoints/decoder_sentencepiece.model"

    # Target languages for Sonar translator
    tgt_langs: List[str] = field(default_factory=lambda: ["eng_Latn"])

    # Generation parameters
    # Note: these are ignored when using SONAR
    num_translations: int = 5
    do_sample: bool = False
    num_beams: int = 5
    temperature: float = 1.0
    max_length: int = 128

    verbose: bool = II("verbose")


@dataclass
class RunConfig:
    video_path: str = MISSING

    preprocessing: PreprocessingConfig = field(default_factory=lambda: PreprocessingConfig())
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig())
    translation: TranslationConfig = field(default_factory=TranslationConfig())
    use_sonar: bool = False
    verbose: bool = False
