# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import time
from pathlib import Path
from typing import Any, Generator

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ssvp_slt.modeling import sign_hiera
from ssvp_slt.modeling.sign_hiera import SignHiera
from ssvp_slt.util.misc import load_model


def shard_generator(data: Any, shard_size: int) -> Generator[Any, None, None]:
    for i in range(0, len(data), shard_size):
        yield data[i : i + shard_size]


class FeatureExtractor:
    def __init__(self, config: DictConfig, device: torch.device):
        self.config = config
        self.device = device

        self.model = self._load_model()

    def _load_model(self) -> SignHiera:
        """
        Loads a pretrained SignHiera model for feature extraction and moves it to specified device
        """

        model = sign_hiera.__dict__[self.config.model_name](pretrained=False, strict=False)

        print("Loading feature extractor")
        load_model(model, Path(self.config.pretrained_model_path))

        model.head = nn.Identity()
        model.eval()
        model.to(self.device)

        return model

    @torch.inference_mode()
    def __call__(self, frames: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:

        t0 = time.time()

        frames = frames.to(self.device)
        padding = padding.to(self.device)

        if len(frames) > self.config.max_batch_size:

            shard_outputs = []

            frame_shards = shard_generator(frames, self.config.max_batch_size)
            padding_shards = shard_generator(padding, self.config.max_batch_size)

            for frames_shard, padding_shard in zip(frame_shards, padding_shards):
                with torch.cuda.amp.autocast(enabled=self.config.fp16):
                    shard_output = self.model.extract_features(frames_shard, padding=padding_shard)
                if len(shard_output.shape) == 1:
                    shard_output = shard_output.unsqueeze(0)

                shard_outputs.append(shard_output)

            outputs = torch.concatenate(shard_outputs, dim=0)
        else:
            with torch.cuda.amp.autocast(enabled=self.config.fp16):
                outputs = self.model.extract_features(frames, padding=padding)

        t1 = time.time()

        if self.config.verbose:
            print(f"3. Feature extraction: {t1 - t0:.3f}s")

        return outputs
