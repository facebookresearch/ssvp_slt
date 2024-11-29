import os
import time
from pathlib import Path
import torch
from omegaconf import DictConfig, OmegaConf
from src.slt_inference.preprocessing import Preprocessor
from src.slt_inference.feature_extraction import FeatureExtractor
from src.slt_inference.translation import SonarTranslator, Translator
from src.slt_inference.util import FLORES200_ID2LANG, print_translations

def e2e_pipeline(config: DictConfig):
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
    preprocessor = Preprocessor(config.preprocessing, device=device)
    feature_extractor = FeatureExtractor(config.feature_extraction, device=device)
    translator_cls = SonarTranslator if config.use_sonar else Translator
    translator = translator_cls(config.translation, device=device, dtype=dtype)
    
    t1 = time.time()
    if config.verbose:
        print(f"1. Model loading: {t1 - t0:.3f}s")
        
    # Process input
    inputs = preprocessor(Path(config.video_path))
    extracted_features = feature_extractor(**inputs)
    
    kwargs = {"tgt_langs": config.translation.tgt_langs} if config.use_sonar else {}
    translations = translator(extracted_features, **kwargs)["translations"]

    keys = [FLORES200_ID2LANG[lang] for lang in config.translation.tgt_langs] if config.use_sonar else range(config.translation.num_translations)

    # Output results
    print_translations(keys, translations)
