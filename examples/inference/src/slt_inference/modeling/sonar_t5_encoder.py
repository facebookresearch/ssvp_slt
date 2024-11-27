from dataclasses import dataclass
from typing import Optional

import torch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import DataType, Device
from omegaconf import DictConfig
from torch import Tensor, nn

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
        padding_mask = batch.padding_mask.materialize() if batch.padding_mask is not None else None

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
    config: DictConfig, *, device: Optional[Device] = None, dtype: Optional[DataType] = None
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
