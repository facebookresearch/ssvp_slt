#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



python scripts/video_feature_extractor/extract.py \
    --vdir <path_to_video_folder> \
    --fdir data/feat/feat_how2_s3d \
    --type=s3d --num_decoding_thread=4 \
    --batch_size 32 --half_precision 1
