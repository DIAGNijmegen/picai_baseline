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

import pickle
from pathlib import Path
from typing import List

import numpy as np
from picai_baseline.nnunet.softmax_export import \
    save_softmax_nifti_from_softmax
from picai_baseline.splits.picai import train_splits, valid_splits
from picai_eval.image_utils import read_prediction
from tqdm import tqdm

nnUNet_results = Path("/workdir/results/nnUNet/3d_fullres/")
in_dir_scans_original = Path("/input/images")
in_dir_scans_cropped = Path("/workdir/")
nnUNet_task = nnUNet_results / "Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1"
out_dir_pred = nnUNet_task / "picai_pubtrain_predictions_model_best_ensemble"
in_dir_pred = [
    nnUNet_task / f"fold_{fold}/picai_pubtrain_predictions_model_best"
    for fold in range(5)
]
in_dir_properties = in_dir_pred[1]
subject_list: List[str] = train_splits[0]["subject_list"] + valid_splits[0]["subject_list"]

out_dir_pred.mkdir(exist_ok=True)

# ensemble model predictions from 5-fold cross-validation
for subject_id in tqdm(subject_list):
    out_path_softmax = out_dir_pred / f"{subject_id}.nii.gz"
    if out_path_softmax.exists():
        pass # continue

    # read predictions from all folds
    preds = [read_prediction(in_dir / f"{subject_id}.npz") for in_dir in in_dir_pred]

    # average predictions
    pred = np.mean(preds, axis=0)

    # convert to nnUNet's format
    pred = np.array([1-pred, pred])

    # read physical properties of current case
    properties_path = in_dir_properties / f"{subject_id}.pkl"
    with open(properties_path, "rb") as fp:
        properties = pickle.load(fp)

    # let nnUNet resample to original physical space
    out_path_softmax = out_dir_pred / f"{subject_id}.nii.gz"
    save_softmax_nifti_from_softmax(
        segmentation_softmax=pred,
        out_fname=str(out_path_softmax),
        properties_dict=properties,
    )

"""
docker run --cpus=4 --memory=32gb --rm \
    -v /path/to/workdir:/workdir/ \
    -v /path/to/repos:/repos/ \
    -v /data/central/pelvis/data/prostate-MRI/picai/public_training/images/:/input/images/ \
    joeranbosma/picai_nnunet:latest python /repos/picai_baseline/src/picai_baseline/nnunet/ensemble_predictions.py
"""
