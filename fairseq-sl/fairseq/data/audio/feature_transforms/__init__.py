# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from fairseq.data.audio import (
    AudioTransform,
    CompositeAudioTransform,
    import_transforms,
    register_audio_transform,
)


class AudioFeatureTransform(AudioTransform):
    pass


AUDIO_FEATURE_TRANSFORM_REGISTRY = {}
AUDIO_FEATURE_TRANSFORM_CLASS_NAMES = set()


def get_audio_feature_transform(name):
    return AUDIO_FEATURE_TRANSFORM_REGISTRY[name]


def register_audio_feature_transform(name):
    return register_audio_transform(
        name,
        AudioFeatureTransform,
        AUDIO_FEATURE_TRANSFORM_REGISTRY,
        AUDIO_FEATURE_TRANSFORM_CLASS_NAMES,
    )


import_transforms(os.path.dirname(__file__), "feature")


class CompositeAudioFeatureTransform(CompositeAudioTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        return super()._from_config_dict(
            cls,
            "feature",
            get_audio_feature_transform,
            CompositeAudioFeatureTransform,
            config,
        )
