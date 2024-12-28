# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import DataType, Device
from omegaconf import DictConfig
from ssvp_slt.util.misc import load_model
from torch import Tensor, nn

from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

from .sign_t5 import SignT5Config, SignT5Model


@dataclass
class SonarEncoderOutput:
    """Dataclass for both speech and text SONAR encoder outputs"""

    encoded_seqs: Tensor
    """Holds the output of the encoder
    *Shape:* :math:`(N,S,M)`, where :math:`N` is the batch size,
    :math:`S` is the sequence length, and :math:`M` is the
    dimensionality of the model.
    """

    sentence_embeddings: Tensor
    """ Pooled representation, derived from encoded_seqs by pooling in dim=1
    *Shape:* :math:`(N,M)`, where :math:`N` is the batch size, and :math:`M` is the
    dimensionality of the model.
    """

    padding_mask: Optional[PaddingMask]
    """Optional, the floating padding mask over sequences (-inf means masked element)
    *Shape:* :math:`(N,S)`, where :math:`N` is the batch size,
    :math:`S` is the sequence length.
    """


class SignT5SonarEncoder(SignT5Model):
    def __init__(self, config: SignT5Config, output_dim: int = 1024):
        super().__init__(config)

        self.projection_out = nn.Linear(config.d_model, output_dim, bias=False)
        self.projection_out.weight.data.normal_(mean=0.0, std=1e-4)

    @property
    def bos_idx(self):
        return self.config.decoder_start_token_id

    def forward(self, batch: SequenceBatch) -> SonarEncoderOutput:

        seqs = batch.seqs
        padding_mask = (
            batch.padding_mask.materialize() if batch.padding_mask is not None else None
        )

        encoded_seqs = self.encoder(
            attention_mask=padding_mask,
            inputs_embeds=seqs,
        )[0]

        decoder_out = self.decoder(
            input_ids=self._get_pooling_tokens(batch.batch_size, seqs.device),
            encoder_hidden_states=encoded_seqs,
            encoder_attention_mask=padding_mask,
        )[0]

        sentence_embeddings = self.projection_out(decoder_out).squeeze(1)

        return SonarEncoderOutput(
            encoded_seqs=encoded_seqs,
            sentence_embeddings=sentence_embeddings,
            padding_mask=batch.padding_mask,
        )

    def _get_pooling_tokens(self, batch_size: int, device: Device) -> Tensor:
        return torch.tensor(
            [self.bos_idx] * batch_size, dtype=torch.int64, device=device
        ).unsqueeze(1)


def create_sonar_signt5_encoder_model(
    config: DictConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
):

    config = SignT5Config.from_pretrained(
        config.base_model_name,
        decoder_start_token_id=0,
        dropout_rate=0.0,
    )
    model = SignT5SonarEncoder(config)

    if device is not None:
        model.to(device)

    if dtype is not None:
        model.to(dtype)

    return model


class SonarTranslator:
    def __init__(
        self,
        config: DictConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype

        self.encoder, self.decoder = self._load_encoder_and_decoder()

    def _load_encoder_and_decoder(
        self,
    ) -> Tuple[nn.Module, EmbeddingToTextModelPipeline]:

        encoder = create_sonar_signt5_encoder_model(
            self.config, device=self.device, dtype=self.dtype
        )
        decoder = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device,
            dtype=self.dtype,
        )

        print("Loading translator")
        load_model(
            encoder, Path(self.config.pretrained_model_path), model_key="student"
        )

        encoder.eval()

        return encoder, decoder

    @torch.inference_mode()
    def __call__(
        self,
        features: torch.Tensor,
        tgt_langs: Sequence[str] = ["eng_Latn"],
    ) -> Dict[str, List[str]]:

        t0 = time.time()

        features = features.to(self.device)

        sentence_embedding = self.encoder(
            SequenceBatch(seqs=features.unsqueeze(0), padding_mask=None)
        ).sentence_embeddings

        translations = []
        for tgt_lang in tgt_langs:
            generated_text = self.decoder.predict(sentence_embedding, tgt_lang)[0]
            assert isinstance(generated_text, str)
            translations.append(generated_text)

        t1 = time.time()

        if self.config.verbose:
            print(f"Translation: {t1 - t0:.3f}s")

        return {"translations": translations}
