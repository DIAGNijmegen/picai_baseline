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
import os
import shutil
import zipfile
from pathlib import Path
from subprocess import check_call

from picai_baseline.splits.picai_pubpriv import nnunet_splits
from picai_baseline.unet.plan_overview import main as plan_overview


def main():
    """Train U-Net semi-supervised model."""
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

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Convert MHA Archive to nnU-Net Raw Data Archive
    # Also, we combine the provided human-expert annotations with the AI-derived annotations.
    print("Preprocessing data...")
    cmd = [
        "python",
        (local_scripts_dir / "picai_baseline/src/picai_baseline/prepare_data_semi_supervised.py").as_posix(),
        "--workdir", workdir.as_posix(),
        "--imagesdir", images_dir.as_posix(),
        "--labelsdir", labels_dir.as_posix(),
        "--spacing", "3.0", "0.5", "0.5",
        "--matrix_size", "20", "256", "256",
    ]
    check_call(cmd)

    # Create UNet overviews
    print("Creating UNet overviews...")
    plan_overview(
        preprocessed_data_path=workdir / 'nnUNet_raw_data/Task2203_picai_baseline/',
        overviews_path=checkpoints_dir / 'results/UNet/overviews/',
        splits=nnunet_splits,
    )

    # Train models
    print(f"Training model...")
    folds = range(5)  # range(5) for 5-fold cross-validation
    cmd = [
        "python", (local_scripts_dir / "picai_baseline/src/picai_baseline/unet/train.py").as_posix(),
        "--weights_dir", (checkpoints_dir / "results/UNet/weights_semisupervised/").as_posix(),
        "--overviews_dir", (checkpoints_dir / "results/UNet/overviews/").as_posix(),
        "--folds", *[str(fold) for fold in folds],
        "--max_threads", "6",
        "--enable_da", "1",
        "--num_epochs", "2",  # 250 for baseline U-Net
        "--validate_n_epochs", "1",
        "--validate_min_epoch", "0",
    ]
    check_call(cmd)

    # Export trained models
    for fold in folds:
        src = checkpoints_dir / f"results/UNet/weights_semisupervised/unet_F{fold}.pt"
        dst = output_dir / f"picai_unet_semi_supervised_gc_algorithm/results/UNet/weights_semisupervised/unet_F{fold}.pt"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    shutil.copy(workdir / "results/UNet/weights_semisupervised/plans.pkl",
                output_dir / "picai_unet_semi_supervised_gc_algorithm/results/UNet/weights_semisupervised/plans.pkl")


if __name__ == '__main__':
    main()
