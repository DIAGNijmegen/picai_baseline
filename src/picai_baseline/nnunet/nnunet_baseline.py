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

import os
import shutil
from pathlib import Path
from subprocess import check_call

from picai_baseline.splits.picai_nnunet import train_splits, valid_splits
from picai_eval import evaluate_folder
from report_guided_annotation import extract_lesion_candidates

"""
nnUNet baseline for the PI-CAI Challenge
For documentation, please see:
https://github.com/DIAGNijmegen/picai_baseline#nnunet-baseline
"""

# environment settings
if 'inputdir' in os.environ:
    inputdir = Path(os.environ['inputdir'])
else:
    inputdir = Path("/media/pelvis/data/prostate-MRI/picai/public_training")
if 'workdir' in os.environ:
    workdir = Path(os.environ['workdir'])
else:
    workdir = Path("/media/pelvis/projects/joeran/picai/workdir")

# settings
task = "Task2201_picai_baseline"
trainer = "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints"

# paths
mha_archive_dir = inputdir / "images"
annotations_dir = inputdir / "labels/csPCa_lesion_delineations/human_expert/resampled/"
mha2nnunet_settings_path = workdir / "mha2nnunet_settings" / f"{task}.json"
nnUNet_raw_data_path = workdir / "nnUNet_raw_data"
nnUNet_task_dir = nnUNet_raw_data_path / task
nnUNet_dataset_json_path = nnUNet_task_dir / "dataset.json"
nnUNet_splits_path = nnUNet_task_dir / "splits.json"
splits_config_root = Path(__file__).parent / "splits"
nnUNet_trainer_results = workdir / "results/nnUNet/3d_fullres/" / task / "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1"
nnUNet_gc_algorithm_results = workdir / "picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/" / task / "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1"

# train nnUNet (first fold will trigger preprocessing as well)
print("Starting training...")
for fold in range(5):
    # check if training already finished for this fold
    if (workdir / f"results/nnUNet/3d_fullres/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_final.model").exists():
        print(f"Training already finished for fold {fold}, skipping..")
    else:
        cmd = [
            "nnunet", "plan_train", task,
            workdir.as_posix(),
            "--trainer", trainer,
            "--custom_split", nnUNet_splits_path.as_posix(),
            "--fold", fold
        ]
        check_call([str(c) for c in cmd])
        print(f"Training finished for fold {fold}.")

# perform inference with nnUNet (predict the full train and validation dataset)
print("Starting inference...")
for fold in range(5):
    predictions_dir = nnUNet_task_dir / f"predictions_fold_{fold}"

    # check if inference already finished for this fold
    if predictions_dir.exists():
        print(f"Inference already finished for fold {fold}, skipping..")
    else:
        cmd = [
            "nnunet", "predict", task,
            "--trainer", trainer,
            "--fold", fold,
            "--checkpoint", "model_best",
            "--results", (workdir / "results").as_posix(),
            "--input", (nnUNet_task_dir / "imagesTr").as_posix(),
            "--output", predictions_dir.as_posix(),
            "--store_probability_maps",
        ]
        check_call([str(c) for c in cmd])
        print(f"Inference finished for fold {fold}.")

# perform evaluation
print("Starting evaluation...")
for fold in range(5):
    predictions_dir = nnUNet_task_dir / f"predictions_fold_{fold}"

    # check if evaluation already finished for this fold
    if (workdir / f"predictions_fold_{fold}/metrics.json").exists():
        print(f"Evaluation already finished for fold {fold}, skipping..")
    else:
        metrics = evaluate_folder(
            y_det_dir=predictions_dir,
            y_true_dir=(nnUNet_task_dir / "labelsTr"),
            subject_list=valid_splits[fold],
            y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
        )
        metrics.save(predictions_dir / "metrics.json")
        print(f"Evaluation finished for fold {fold}.")

    # optionally, evaluate training performance
    if (workdir / f"predictions_fold_{fold}/metrics-train.json").exists():
        print(f"Evaluation of training performance already finished for fold {fold}, skipping..")
    else:
        metrics = evaluate_folder(
            y_det_dir=predictions_dir,
            y_true_dir=(nnUNet_task_dir / "labelsTr"),
            subject_list=train_splits[fold],
            y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
        )
        metrics.save(predictions_dir / "metrics-train.json")
        print(f"Evaluation of training performance finished for fold {fold}.")

# gather resources for inference deployment
for path in [
    "fold_0/model_best.model",
    "fold_0/model_best.model.pkl",
    "fold_1/model_best.model",
    "fold_1/model_best.model.pkl",
    "fold_2/model_best.model",
    "fold_2/model_best.model.pkl",
    "fold_3/model_best.model",
    "fold_3/model_best.model.pkl",
    "fold_4/model_best.model",
    "fold_4/model_best.model.pkl",
    "plans.pkl",
]:
    shutil.copyfile(
        nnUNet_trainer_results / path,
        nnUNet_gc_algorithm_results / path,
    )
