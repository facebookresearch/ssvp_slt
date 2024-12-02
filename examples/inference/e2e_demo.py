from omegaconf import OmegaConf
from run import e2e_pipeline
from src.slt_inference.configs import RunConfig, FeatureExtractionConfig, TranslationConfig

# Define the translation configuration
translation_config = RunConfig(
    video_path="./video.mp4",
    verbose=True,
    use_sonar=False,
    feature_extraction=FeatureExtractionConfig(
        pretrained_model_path="./signhiera_mock.pth",     
    ),
    translation=TranslationConfig(
        base_model_name="google-t5/t5-base",
        tgt_langs=["eng_Latn", "fra_Latn"]
    )
)

# Convert it to DictConfig
translation_dict_config = OmegaConf.structured(translation_config)

# Run pipeline with provided parameters
e2e_pipeline(translation_dict_config)
