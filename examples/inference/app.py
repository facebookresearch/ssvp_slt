import os
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from time import sleep
from typing import List, Optional, Sequence, Tuple

import gradio as gr
import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import II, DictConfig, OmegaConf

from slt_inference.feature_extraction import FeatureExtractor
from slt_inference.preprocessing import Preprocessor
from slt_inference.translation import SonarTranslator, Translator
from slt_inference.util import FLORES200_ID2LANG, print_translations


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
    tokenizer_path: str = "checkpoints/tokenizer"
    base_model_name: str = "google/t5-v1_1-large"
    feature_dim: int = 768
    decoder_path: str = "checkpoints/sonar_decoder.pt"
    decoder_spm_path: str = "checkpoints/decoder_sentencepiece.model"

    # Target languages for Sonar translator
    tgt_langs: List[str] = field(default_factory=lambda: ["eng_Latn"])

    # Generation parameters
    num_translations: int = 5
    do_sample: bool = False
    num_beams: int = 5
    temperature: float = 1.0
    max_length: int = 128

    verbose: bool = II("verbose")

    def __post_init__(self):
        for lang in self.tgt_langs:
            if lang not in FLORES200_ID2LANG:
                raise ValueError(f"{lang} is not a valid FLORES-200 language ID")


@dataclass
class RunConfig:
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    feature_extraction: FeatureExtractionConfig = FeatureExtractionConfig()
    translation: TranslationConfig = TranslationConfig()
    use_sonar: bool = False
    verbose: bool = False


cs = ConfigStore.instance()
cs.store(name="run_config", node=RunConfig)

video = None
video_released = False
translation_queue = Queue(maxsize=1)

css = """
.app {
    max-width: 50% !important;
}
"""


@hydra.main(config_name="run_config")
def main(config: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    print(f"Config:\n{OmegaConf.to_yaml(config, resolve=True)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if config.feature_extraction.fp16 else torch.float32

    preprocessor = Preprocessor(config.preprocessing, device=device)
    feature_extractor = FeatureExtractor(config.feature_extraction, device=device)

    translator_cls = SonarTranslator if config.use_sonar else Translator
    translator = translator_cls(config.translation, device=device, dtype=dtype)

    def release():
        """Triggered when video ends to indicate translation can now be displayed"""
        global video_released
        video_released = True

    def load_video(video_path: str):
        """Load and preprocess the uploaded video"""
        global video
        video = preprocessor(Path(video_path))

    def inference_video():
        """Run inference on video and put translations in a queue"""
        global translation_queue
        extracted_features = feature_extractor(**video)

        kwargs = {"tgt_langs": config.translation.tgt_langs} if config.use_sonar else {}
        translations = translator(extracted_features, **kwargs)["translations"]
        translation_queue.put(translations)

        keys = (
            [FLORES200_ID2LANG[lang] for lang in config.translation.tgt_langs]
            if config.use_sonar
            else range(config.translation.num_translations)
        )
        print_translations(keys, translations)

    def show_translation():
        """Consume translation results from queue and display them"""

        global video_released
        # Wait until video has finished playing before showing translation
        while not video_released:
            sleep(0.05)

        video_released = False
        translation_result = translation_queue.get()

        return (
            [gr.Text(v) for v in translation_result]
            if config.use_sonar
            else gr.Text(translation_result[0])
        )

    with gr.Blocks(css=css) as demo:
        with gr.Column(scale=1):
            gr.Markdown("### ASL")
            input_video = gr.Video(label="Input Video", height=360, autoplay=True)

            output_texts = []
            if config.use_sonar:
                for lang in config.translation.tgt_langs:
                    gr.Markdown(f"### {FLORES200_ID2LANG[lang]}")
                    output_texts.append(gr.Text("", interactive=False, label="Translation"))
            else:
                gr.Markdown("### English")
                output_texts.append(gr.Text("", interactive=False, label="Translation"))

        input_video.upload(fn=load_video, inputs=input_video, outputs=None).success(
            fn=inference_video, inputs=None, outputs=None
        ).success(fn=show_translation, inputs=None, outputs=output_texts)

        input_video.end(fn=release, inputs=None, outputs=None)

    demo.launch()


if __name__ == "__main__":
    main()
