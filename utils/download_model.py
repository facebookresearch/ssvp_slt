# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import wget

def get_model_path(url: str,  model_path: str = 'signhiera_mock.pth'):
     # Check if the model file exists
    if os.path.exists(model_path):
        print(f"Model already exists at: {model_path}")
    else:
        print("Model not found, downloading...")
        filename = wget.download(url, model_path)
        print(f"Downloaded model to: {filename}")
    
    return model_path