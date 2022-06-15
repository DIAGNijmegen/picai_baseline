#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from pathlib import Path


# read dataset configurations
train_splits = {}
for fold in range(5):
    with open(Path(__file__).parent / f"ds-config-train-fold-{fold}.json") as fp:
        ds_config = json.load(fp)
        train_splits[fold] = ds_config

valid_splits = {}
for fold in range(5):
    with open(Path(__file__).parent / f"ds-config-valid-fold-{fold}.json") as fp:
        ds_config = json.load(fp)
        valid_splits[fold] = ds_config

with open(Path(__file__).parent / "splits.json") as fp:
    nnunet_splits = json.load(fp)

# expose dataset configurations
__all__ = [
    "train_splits",
    "valid_splits",
    "nnunet_splits",
]
