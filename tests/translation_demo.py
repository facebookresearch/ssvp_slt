from translation.run_translation_module import Config, run_translation, ModelConfig, DataConfig
from omegaconf import OmegaConf

translation_config = Config(
    data=DataConfig(     
        val_data_dir="features_outputs\\0"
    ),
    model=ModelConfig(
        name_or_path="google-t5/t5-base",
        feature_dim=1024
    )
)

# Convert it to DictConfig
translation_dict_config = OmegaConf.structured(translation_config)

# Run translation with the DictConfig instance
run_translation(translation_dict_config)