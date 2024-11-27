import os
from pathlib import Path
from typing import Union

import numpy as np
import sentencepiece as spm
import torch
from fairseq.models import FairseqEncoder
from fairseq.models.fairseq_model import FairseqEncoderDecoderModel
from fairseq.models.transformer.transformer_decoder import TransformerDecoderBase
from fairseq.sequence_generator import SequenceGenerator
from torch import device


class DummyEncoder(FairseqEncoder):
    def __init__(self):
        super().__init__(None)

    def forward_torchscript(self, net_input):
        return {"encoder_out": [net_input["source"].unsqueeze(0)], "encoder_padding_mask": [None]}

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        encoder_out_dict["encoder_out"][0] = encoder_out_dict["encoder_out"][0].index_select(
            1, new_order
        )
        return encoder_out_dict


class TmodulesTextDecoder:
    def __init__(
        self,
        path_decoder,
        path_spm,
        beam_size=5,
        len_penalty=1.0,
        temperature=1.0,
        no_repeat_ngram_size=0,
        max_len=100,
        device=torch.device("cpu"),
    ):
        self.path_decoder = path_decoder

        checkpoint = torch.load(Path(path_decoder).resolve())
        self.decoder = TransformerDecoderBase(
            checkpoint["cfg"], checkpoint["dictionary"], checkpoint["embed_tokens"]
        )
        self.decoder.load_state_dict(checkpoint["state_dict"])
        self.decoder.eval().to(device)
        self.spm_out = spm.SentencePieceProcessor(model_file=path_spm)
        self.dict_out = checkpoint["dictionary"]
        self.device = device
        self.dummy_encoder = DummyEncoder()
        embedding_model = FairseqEncoderDecoderModel(self.dummy_encoder, self.decoder)
        self.generator = SequenceGenerator(
            [embedding_model],
            self.dict_out,
            beam_size=beam_size,
            len_penalty=len_penalty,
            temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_len=max_len,
        )

    def decode_file(self, file_path, lang="eng_Latn"):
        nbex = int(os.path.getsize(file_path) / 1024 / 2)
        embeddings = np.array(np.memmap(file_path, mode="r", dtype=np.float16, shape=(nbex, 1024)))
        return self.decode(embeddings, lang=lang)

    def decode(self, embeddings, lang="eng_Latn", bz=100):
        if isinstance(embeddings, (np.ndarray, np.generic)):
            embeddings = torch.FloatTensor(embeddings)

        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)

        batches = torch.split(embeddings, bz)
        preds = []

        for batch in batches:
            sample = {
                "net_input": {
                    "source": batch.to(self.device),
                    "padding_mask": torch.tensor([[False]] * batch.shape[0]).to(self.device),
                }
            }
            prefix_tokens = (
                torch.LongTensor([[self.dict_out.index(f"__{lang}__")]])
                .expand(batch.shape[0], 1)
                .to(self.device)
            )
            preds = preds + self.generator.forward(sample, prefix_tokens=prefix_tokens)

        gens = []
        for i in range(len(preds)):
            gens.append(
                self.spm_out.decode_pieces(
                    self.dict_out.string(preds[i][0]["tokens"][1:]).split(" ")
                ).replace("<MINED_DATA> ", "")
            )
        return gens


def load_decoder(
    model_path: str, spm_path: str, *, freeze: bool = True, device: Union[str, device] = "cpu"
):
    model = TmodulesTextDecoder(
        path_decoder=model_path, path_spm=spm_path, max_len=200, device=device
    )

    if freeze:
        for p in model.decoder.parameters():
            p.requires_grad = False

    return model
