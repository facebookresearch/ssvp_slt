# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

grep "seg id" | sed 's/<seg id="[0-9]\+">//g' | sed 's/<\/seg>//g'
