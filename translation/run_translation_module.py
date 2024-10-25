import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import II, MISSING, DictConfig, OmegaConf
from ssvp_slt.util.misc import reformat_logger
from main_translation import main as translate

logger = logging.getLogger(__name__)

# Configuration classes for the module
@dataclass
class CommonConfig:
    output_dir: str = "./translation_output"
    log_dir: str = "./translation_logs"
    resume: Optional[str] = None
    load_model: Optional[str] = None
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    eval: bool = False
    dist_eval: bool = True
    pin_mem: bool = True
    num_workers: int = 10
    eval_print_samples: bool = False
    max_checkpoints: int = 3
    eval_steps: Optional[int] = None
    eval_best_model_after_training: bool = True
    overwrite_output_dir: bool = False
    compute_bleurt: bool = False

@dataclass
class ModelConfig:
    name_or_path: str = None
    feature_dim: int = 512
    from_scratch: bool = False
    dropout: float = 0.3
    num_beams: int = 5
    lower_case: bool = False
    
    # Fairseq-specific fields for model compatibility
    min_source_positions: int = 0
    max_source_positions: int = 1024
    max_target_positions: int = 1024
    feats_type: Optional[str] = "hiera"
    activation_fn: Optional[str] = "relu"
    encoder_normalize_before: Optional[bool] = True
    encoder_embed_dim: Optional[int] = 768
    encoder_ffn_embed_dim: Optional[int] = 3072
    encoder_attention_heads: Optional[int] = 12
    encoder_layerdrop: Optional[float] = 0.1
    encoder_layers: Optional[int] = 12
    decoder_normalize_before: Optional[bool] = True
    decoder_embed_dim: Optional[int] = 768
    decoder_ffn_embed_dim: Optional[int] = 3072
    decoder_attention_heads: Optional[int] = 12
    decoder_layerdrop: Optional[float] = 0.1
    decoder_layers: Optional[int] = 12
    decoder_output_dim: Optional[int] = 768
    classifier_dropout: Optional[float] = 0.1
    attention_dropout: Optional[float] = 0.1
    activation_dropout: Optional[float] = 0.1
    layernorm_embedding: Optional[bool] = True
    no_scale_embedding: Optional[bool] = False
    share_decoder_input_output_embed: Optional[bool] = True
    num_hidden_layers: Optional[int] = 12

@dataclass
class DataConfig:
    train_data_dirs: str = MISSING
    val_data_dir: str = MISSING
    num_epochs_extracted: int = 1
    min_source_positions: int = 0
    max_source_positions: int = 1024
    max_target_positions: int = 1024

@dataclass
class CriterionConfig:
    label_smoothing: float = 0.2

@dataclass
class OptimizationConfig:
    clip_grad: float = 1.0
    lr: float = 0.001
    min_lr: float = 1e-4
    weight_decay: float = 1e-1
    start_epoch: int = 0
    epochs: int = 200
    warmup_epochs: int = 10
    train_batch_size: int = 32
    val_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    early_stopping: bool = True
    patience: int = 10
    epoch_offset: Optional[int] = 0 

@dataclass
class WandbConfig:
    enabled: bool = True
    project: Optional[str] = None
    entity: Optional[str] = None
    name: Optional[str] = None
    run_id: Optional[str] = None
    log_code: bool = True

@dataclass
class DistConfig:
    world_size: int = 1
    port: int = 1
    local_rank: int = -1
    enabled: bool = False
    rank: Optional[int] = None
    dist_url: Optional[str] = None
    gpu: Optional[int] = None
    dist_backend: Optional[str] = None

@dataclass
class Config:
    common: CommonConfig = CommonConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    criterion: CriterionConfig = CriterionConfig()
    optim: OptimizationConfig = OptimizationConfig()
    dist: DistConfig = DistConfig()
    wandb: WandbConfig = WandbConfig()
    debug: bool = False
    fairseq: bool = False

def run_translation(cfg: DictConfig):
    # Process configuration and start translation
    OmegaConf.resolve(cfg)
    reformat_logger()

    if cfg.debug:
        print("Running in debug mode")

    # If evaluating without training
    if cfg.common.eval:
        if cfg.common.load_model is None:
            raise RuntimeError("Evaluation mode requires a specified model.")
        cfg.common.output_dir = os.path.dirname(cfg.common.load_model)
        cfg.common.log_dir = os.path.join(cfg.common.output_dir, "logs")
    else:
        Path.mkdir(Path(cfg.common.output_dir), parents=True, exist_ok=True)
        Path.mkdir(Path(cfg.common.log_dir), parents=True, exist_ok=True)

    translate(cfg)
