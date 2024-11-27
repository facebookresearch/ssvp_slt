import os
import time
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from slt_inference.feature_extraction import FeatureExtractor
from slt_inference.translation import SonarTranslator, Translator
from slt_inference.util import FLORES200_ID2LANG, print_translations
from slt_inference.configs import FeatureExtractor, RunConfig
from slt_inference.preprocessing import Preprocessor

from omegaconf import DictConfig, OmegaConf


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

    translator_cls = SonarTranslator if config.use_sonar else Translator
    translator = translator_cls(config.translation, device=device, dtype=dtype)

    t1 = time.time()

    if config.verbose:
        print(f"1. Model loading: {t1 - t0:.3f}s")

    inputs = preprocessor(Path(config.video_path))
    extracted_features = feature_extractor(**inputs)

    kwargs = {"tgt_langs": config.translation.tgt_langs} if config.use_sonar else {}
    translations = translator(extracted_features, **kwargs)["translations"]

    keys = (
        [FLORES200_ID2LANG[lang] for lang in config.translation.tgt_langs]
        if config.use_sonar
        else range(config.translation.num_translations)
    )
    print_translations(keys, translations)


if __name__ == "__main__":
    main()
