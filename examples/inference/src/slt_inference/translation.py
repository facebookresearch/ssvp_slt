import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from fairseq2.models.sequence import SequenceBatch
from omegaconf import DictConfig
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .modeling.sign_t5 import SignT5Config, SignT5ForConditionalGeneration
from .modeling.sonar_decoder import load_decoder
from .modeling.sonar_t5_encoder import create_sonar_signt5_encoder_model
from .util import load_model


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
        load_model(model, Path(self.config.pretrained_model_path))

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


class SonarTranslator:
    def __init__(
        self,
        config: DictConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype

        self.encoder, self.decoder = self._load_encoder_and_decoder()

    def _load_encoder_and_decoder(
        self,
    ) -> Tuple[nn.Module, nn.Module]:

        encoder = create_sonar_signt5_encoder_model(
            self.config, device=self.device, dtype=self.dtype
        )
        decoder = load_decoder(
            self.config.decoder_path, self.config.decoder_spm_path, device=self.device
        )
        decoder.decoder.to(self.dtype)

        print("Loading translator")
        load_model(encoder, Path(self.config.pretrained_model_path), model_key="student")

        encoder.eval()
        decoder.decoder.eval()

        return encoder, decoder

    @torch.inference_mode()
    def __call__(
        self,
        features: torch.Tensor,
        tgt_langs: Sequence[str] = ["eng_Latn"],
    ) -> Dict[str, str]:

        t0 = time.time()

        features = features.to(self.device)

        sentence_embedding = self.encoder(
            SequenceBatch(seqs=features.unsqueeze(0), padding_mask=None)
        ).sentence_embeddings

        translations = []
        for tgt_lang in tgt_langs:
            generated_text = self.decoder.decode(sentence_embedding, tgt_lang)[0]
            translations.append(generated_text)

        t1 = time.time()

        if self.config.verbose:
            print(f"4. Translation: {t1 - t0:.3f}s")

        return {"translations": translations}
