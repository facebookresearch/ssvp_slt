# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import torch
import dlib

# Check if CUDA is available and if it's using a GPU
if torch.cuda.is_available():
  print("CUDA is available!")
  print(torch.__version__)
  print(torch.version.cuda)
  print(f"Device count: {torch.cuda.device_count()}")
  print(f"Current device: {torch.cuda.current_device()}")
  print(f"Device name: {torch.cuda.get_device_name(0)}")

else:
  print("CUDA is not available.")
  print("Consider using a runtime with GPU acceleration in Colab.")


# If dlib CUDA is available, print details
if dlib.DLIB_USE_CUDA:
  print("dlib CUDA is available!")
  print(f"Number of dlib CUDA devices: {dlib.cuda.get_num_devices()}")
else:
  print("dlib CUDA is not available.")
