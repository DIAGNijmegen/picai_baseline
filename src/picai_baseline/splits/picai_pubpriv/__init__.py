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

nnunet_splits = [
    {
        'train': train_splits[fold]["subject_list"],
        'val': valid_splits[fold]["subject_list"],
    }
    for fold in range(5)
]

# expose dataset configurations
__all__ = [
    "train_splits",
    "valid_splits",
    "nnunet_splits",
]


"""
These splits were generated using the following code:

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def get_patient_list(subject_list, unique=False):
    patient_list = [s.split("_")[0] for s in subject_list]
    if unique:
        patient_list = list(set(patient_list))
    return patient_list

# settings
folds = range(5)
marksheet_path = "/path/to/pubpriv_training/v1/labels/clinical_information/marksheet.csv"

# read marksheet
marksheet = pd.read_csv(marksheet_path, dtype=str)

# list of studies for training and validation
subject_list_trainval = marksheet.apply(lambda row: f"{row.patient_id}_{row.study_id}", axis=1)
patient_list_trainval = get_patient_list(subject_list_trainval, unique=True)
print(f"Have {len(subject_list_trainval)} studies for training and validation" +
      f" ({len(patient_list_trainval)} patients)")

# sort lists for reproducibility
patient_list_trainval = np.array(sorted(list(patient_list_trainval)))

# placeholders
train_subject_list_map = {fold: [] for fold in folds}
valid_subject_list_map = {fold: [] for fold in folds}

# make splits
kf = KFold(n_splits=len(folds), random_state=576, shuffle=True)
for fold, (_, test_indices) in enumerate(kf.split(patient_list_trainval)):
    patient_ids_valid = patient_list_trainval[test_indices]

    # add to validation set
    valid_subject_list = [s for s in subject_list_trainval
                          if s.split("_")[0] in patient_ids_valid]
    valid_subject_list_map[fold] = valid_subject_list

    # all other studies go in the training set
    train_subject_list = [s for s in subject_list_trainval
                          if not s in valid_subject_list]
    train_subject_list_map[fold] = train_subject_list
"""
