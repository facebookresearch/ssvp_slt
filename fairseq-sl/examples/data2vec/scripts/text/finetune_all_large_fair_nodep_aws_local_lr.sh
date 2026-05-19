#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


set -eu

dir="$1"

echo "dir: $dir"

mkdir -p "$dir/log"
sbatch_args="-p wav2vec --nodes=1 --ntasks-per-node=1"
sbatch_args="$sbatch_args --gpus-per-node=1 --cpus-per-task=8 --mem=0 --time=24:00:00"
sbatch_args="$sbatch_args -o $dir/log/decode_sweep_%A.out"
sbatch_args="$sbatch_args -e $dir/log/decode_sweep_%A.err"

sbatch $sbatch_args examples/data2vec/scripts/text/finetune_all_large_fair_local_lr.sh $dir
