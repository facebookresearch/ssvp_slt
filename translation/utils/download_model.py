# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import wget
from pathlib import Path

def download_model(url: str, save_dir: str = "."):
    model_name = Path(url).name
    model_path = Path(save_dir) / model_name
    if model_path.exists():
        print(f"Model already exists at: {model_path}")
        return model_path
    else:
        print("Model not found, downloading...")
        filename = wget.download(url, str(model_path))
        print(f"Downloaded model to: {filename}")
        return filename

