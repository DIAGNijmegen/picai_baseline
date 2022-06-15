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

from picai_baseline.splits import subject_list_annotated
from picai_baseline.splits.picai import (nnunet_splits, train_splits,
                                         valid_splits)

# read dataset configurations
for fold, split in train_splits.items():
    split['subject_list'] = [
        subject_id for subject_id in split['subject_list']
        if subject_id in subject_list_annotated
    ]

for fold, split in valid_splits.items():
    split['subject_list'] = [
        subject_id for subject_id in split['subject_list']
        if subject_id in subject_list_annotated
    ]

for split in nnunet_splits:
    split['train'] = [subject_id for subject_id in split['train'] if subject_id in subject_list_annotated]
    split['val'] = [subject_id for subject_id in split['val'] if subject_id in subject_list_annotated]

# expose dataset configurations
__all__ = [
    "train_splits",
    "valid_splits",
    "nnunet_splits",
]
