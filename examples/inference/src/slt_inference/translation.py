import time
from typing import Dict, Tuple

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .modeling.sign_t5 import SignT5Config, SignT5ForConditionalGeneration

class Translator:
    def __init__(
        self,
        config: DictConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype

        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(
        self,
    ) -> Tuple[SignT5ForConditionalGeneration, PreTrainedTokenizerFast]:
        """
        Loads a pretrained SignT5 model for translation and moves it to specified device
        """

        config = SignT5Config(
            decoder_start_token_id=0,
            output_past=True,
            tie_word_embeddings=False,
            feature_dim=self.config.feature_dim,
        )
        model = SignT5ForConditionalGeneration._from_config(config)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path, use_fast=False, legacy=True
        )

        print("Loading translator")

        model.eval()
        model.to(self.device)
        model.to(self.dtype)

        return model, tokenizer

    @torch.inference_mode()
    def __call__(self, features: torch.Tensor) -> Dict[str, str]:

        t0 = time.time()

        features = features.to(self.device)

        generated_tokens = self.model.generate(
            inputs_embeds=features.unsqueeze(0),
            num_return_sequences=self.config.num_translations,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
        )
        generated_text = [
            t.strip()
            for t in self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        ]

        t1 = time.time()

        if self.config.verbose:
            print(f"4. Translation: {t1 - t0:.3f}s")

        return {"translations": generated_text}