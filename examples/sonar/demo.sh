#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

DLIB_DETECTOR_PATH=$1

if [ -z $DLIB_DETECTOR_PATH ]; then 
    echo "Please provide a local path to the appropriate face detector model."
    exit 1
fi 

DOWNLOAD_CACHE=$HOME/.cache/asl
BASE_URL=https://dl.fbaipublicfiles.com/SONAR/asl

if [ ! -d $DOWNLOAD_CACHE ]; then mkdir -p $DOWNLOAD_CACHE; fi 

function download {
    file_name=$1
    if [ -f ${DOWNLOAD_CACHE}/${file_name} ] ; then
        echo " - $file_name already downloaded";
    else
        echo " - Downloading ${file_name}";
        wget -q $BASE_URL/${file_name} -P $DOWNLOAD_CACHE;
    fi 
}

# step 1: Download a sample video
INPUT_VIDEO=0043626-2023.1.4.mp4
download $INPUT_VIDEO

# step 2: Download the feature extractor
FEATURE_EXTRACTOR=dm_70h_ub_signhiera.pth
download $FEATURE_EXTRACTOR

# step 3: Download the sonar student encoder
SONAR_ENCODER=dm_70h_ub_sonar_encoder.pth
download $SONAR_ENCODER

# step 4: Select the target languages for translation from FLORES200 (https://github.com/facebookresearch/flores/blob/main/flores200/README.md)
TGT_LANGS='[eng_Latn, fra_Latn, deu_Latn, zho_Hans]'

python run.py \
    video_path=$DOWNLOAD_CACHE/$INPUT_VIDEO \
    preprocessing.detector_path=$DLIB_DETECTOR_PATH \
    feature_extraction.pretrained_model_path=$DOWNLOAD_CACHE/$FEATURE_EXTRACTOR \
    translation.pretrained_model_path=$DOWNLOAD_CACHE/$SONAR_ENCODER \
    translation.tgt_langs="$TGT_LANGS"
