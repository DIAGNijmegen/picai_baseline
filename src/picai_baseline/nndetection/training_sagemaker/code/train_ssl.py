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

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from subprocess import check_call

from picai_baseline.splits.picai import nnunet_splits


def main(taskname="Task2203_picai_baseline"):
    """Train nnDetection semi-supervised model."""
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--workdir', type=str, default="/workdir")
    parser.add_argument('--imagesdir', type=str, default=os.environ.get('SM_CHANNEL_IMAGES', "/input/images"))
    parser.add_argument('--labelsdir', type=str, default=os.environ.get('SM_CHANNEL_LABELS', "/input/picai_labels"))
    parser.add_argument('--scriptsdir', type=str, default=os.environ.get('SM_CHANNEL_SCRIPTS', "/scripts"))
    parser.add_argument('--outputdir', type=str, default=os.environ.get('SM_MODEL_DIR', "/output"))
    parser.add_argument('--checkpointsdir', type=str, default="/checkpoints")

    args, _ = parser.parse_known_args()

    # paths
    workdir = Path(args.workdir)
    images_dir = Path(args.imagesdir)
    labels_dir = Path(args.labelsdir)
    output_dir = Path(args.outputdir)
    scripts_dir = Path(args.scriptsdir)
    checkpoints_dir = Path(args.checkpointsdir)
    splits_path = workdir / f"splits/{taskname}/splits.json"

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_path.parent.mkdir(parents=True, exist_ok=True)

    # set environment variables
    os.environ["det_data"] = str(workdir / "nnDet_data")

    # extract scripts
    with zipfile.ZipFile(scripts_dir / "code.zip", 'r') as zf:
        zf.extractall(workdir)
    local_scripts_dir = workdir / "code"

    # descibe input data
    print(f"workdir: {workdir}")
    print(f"images_dir: {images_dir}")
    print(f"labels_dir: {labels_dir}")
    print(f"output_dir: {output_dir}")
    print(f"scripts_dir: {local_scripts_dir}")

    print("Scripts folder:", os.listdir(local_scripts_dir))
    print("Images folder:", os.listdir(images_dir))
    print("Labels folder:", os.listdir(labels_dir))

    # save cross-validation splits to disk
    with open(splits_path, "w") as fp:
        json.dump(nnunet_splits, fp)

    # Convert MHA Archive to nnU-Net Raw Data Archive
    # Also, we combine the provided human-expert annotations with the AI-derived annotations.
    print("Preprocessing data...")
    cmd = [
        "python",
        (local_scripts_dir / "picai_baseline/src/picai_baseline/prepare_data_semi_supervised.py").as_posix(),
        "--workdir", workdir.as_posix(),
        "--imagesdir", images_dir.as_posix(),
        "--labelsdir", labels_dir.as_posix(),
    ]
    check_call(cmd)

    # Convert nnU-Net Raw Data Archive to nnDetection Raw Data Archive
    print("Converting data...")
    cmd = [
        "python", "-m", "picai_prep", "nnunet2nndet",
        "--input", (workdir / "nnUNet_raw_data/Task2203_picai_baseline").as_posix(),
        "--output", (workdir / f"nnDet_raw_data/{taskname}").as_posix(),
    ]

    # Train models
    folds = range(1)  # range(5) for 5-fold cross-validation
    for fold in folds:
        print(f"Training fold {fold}...")
        cmd = [
            "nndet",
            "prep_train",
            str(taskname),
            workdir.as_posix(),
            "--results", checkpoints_dir.as_posix(),
            "--custom_split", str(splits_path),
            "--fold", str(fold),
        ]
        check_call(cmd)

    # Export trained models
    output_task_dir = output_dir / "picai_nndetection_gc_algorithm/results/nnDet" / taskname
    consolidated_model_dir = output_task_dir / "RetinaUNetV001_D3V001_3d/consolidated"
    consolidated_model_dir.mkdir(parents=True, exist_ok=True)
    input_consolidated_model_dir = checkpoints_dir / f"nnDet/{taskname}/RetinaUNetV001_D3V001_3d/consolidated"
    for fold in folds:
        src = checkpoints_dir / f"nnDet/{taskname}/RetinaUNetV001_D3V001_3d/consolidated/model_fold{fold}.ckpt"
        dst = consolidated_model_dir / f"model_fold{fold}.ckpt"
        shutil.copy(src, dst)

    shutil.copy(
        input_consolidated_model_dir / "config.yaml",
        consolidated_model_dir / "config.yaml",
    )
    shutil.copy(
        input_consolidated_model_dir / "plan_inference.pkl",
        consolidated_model_dir / "plan_inference.pkl",
    )
    shutil.copy(
        workdir / f"nnDet_raw_data/{taskname}/dataset.json",
        output_task_dir / "dataset.json",
    )

if __name__ == '__main__':
    main()
