from omegaconf import OmegaConf
from e2e import e2e_pipeline
from src.slt_inference.configs import RunConfig, FeatureExtractionConfig, TranslationConfig

# Define the translation configuration
translation_config = RunConfig(
    video_path="D:/Pro/MLH/ctf/video.mp4",
    verbose=True,
    feature_extraction=FeatureExtractionConfig(
        pretrained_model_path="translation/signhiera_mock.pth",     
    ),
    translation=TranslationConfig(
        base_model_name="google/t5-v1_1-large",
        tgt_langs=["eng_Latn", "fra_Latn"]
    )
)

# Convert it to DictConfig
translation_dict_config = OmegaConf.structured(translation_config)

# Run pipeline with provided parameters
e2e_pipeline(translation_dict_config)
