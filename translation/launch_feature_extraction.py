# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import asyncio
import hydra
from feature_extraction_module import FeatureExtractionConfig, FeatureExtractionModule
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from stopes.core import Launcher

async def run(config: FeatureExtractionConfig):
    launcher = Launcher(
        cluster=config.launcher.cluster,
        partition=config.launcher.partition,
        max_jobarray_jobs=config.launcher.max_jobarray_jobs,
    )
    module = FeatureExtractionModule(config)
    await launcher.schedule(module)


cs = ConfigStore.instance()
cs.store(name="feature_extraction_config", node=FeatureExtractionConfig)


@hydra.main(version_base=None, config_name="feature_extraction_config")
def main(config: DictConfig):
    OmegaConf.resolve(config)

    print(f"Config:\n{'-'*50}\n{OmegaConf.to_yaml(config)}{'-'*50}")

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
