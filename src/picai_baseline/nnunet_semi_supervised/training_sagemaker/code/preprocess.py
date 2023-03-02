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
from pathlib import Path
from subprocess import check_call

from picai_baseline.prepare_data_semi_supervised import \
    prepare_data_semi_supervised
from picai_baseline.splits.picai import nnunet_splits as picai_pub_splits
from picai_baseline.splits.picai_debug import \
    nnunet_splits as picai_debug_splits
from picai_baseline.splits.picai_nnunet import \
    nnunet_splits as picai_pub_nnunet_splits
from picai_baseline.splits.picai_pubpriv import \
    nnunet_splits as picai_pubpriv_splits
from picai_baseline.splits.picai_pubpriv_nnunet import \
    nnunet_splits as picai_pubpriv_nnunet_splits


def main(taskname="Task2203_picai_baseline"):
    """Preprocess data for nnU-Net semi-supervised model."""
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--workdir', type=str, default="/workdir")
    parser.add_argument('--imagesdir', type=str, default=os.environ.get('SM_CHANNEL_IMAGES', "/input/images"))
    parser.add_argument('--labelsdir', type=str, default=os.environ.get('SM_CHANNEL_LABELS', "/input/picai_labels"))
    parser.add_argument('--outputdir', type=str, default=os.environ.get('SM_MODEL_DIR', "/output"))
    parser.add_argument('--splits', type=str, default="picai_pubpriv",
                        help="Cross-validation splits. Can be a path to a json file or one of the predefined splits: "
                             "picai_pub, picai_pubpriv, picai_pub_nnunet, picai_pubpriv_nnunet, picai_debug.")
    parser.add_argument('--nnUNet_tf', type=int, default=8, help="Number of preprocessing threads for full images")
    parser.add_argument('--nnUNet_tl', type=int, default=8, help="Number of preprocessing threads for low-res images")

    args, _ = parser.parse_known_args()

    # paths
    workdir = Path(args.workdir)
    images_dir = Path(args.imagesdir)
    labels_dir = Path(args.labelsdir)
    output_dir = Path(args.outputdir)
    splits_path = workdir / f"nnUNet_raw_data/{taskname}/splits.json"
    nnUNet_prep_dir = output_dir / "nnUNet_preprocessed"

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # set environment variables
    os.environ["prepdir"] = str(nnUNet_prep_dir)

    # set nnU-Net's number of preprocessing threads
    os.environ["nnUNet_tf"] = str(args.nnUNet_tf)
    os.environ["nnUNet_tl"] = str(args.nnUNet_tl)

    # descibe input data
    print(f"workdir: {workdir}")
    print(f"images_dir: {images_dir}")
    print(f"labels_dir: {labels_dir}")
    print(f"output_dir: {output_dir}")

    print("Images folder:", os.listdir(images_dir))
    print("Labels folder:", os.listdir(labels_dir))

    # Convert MHA Archive to nnU-Net Raw Data Archive
    # Also, we combine the provided human-expert annotations with the AI-derived annotations.
    print("Preprocessing data...")
    prepare_data_semi_supervised(
        workdir=workdir,
        imagesdir=images_dir,
        labelsdir=labels_dir,
        preprocessing_kwargs='{"physical_size": [81.0, 192.0, 192.0], "crop_only": true}',
        splits=args.splits,
    )

    # Preprocess data with nnU-Net
    print("Preprocessing data with nnU-Net...")
    cmd = [
        "nnunet", "plan_train", str(taskname), workdir.as_posix(),
        "--custom_split", str(splits_path),
        "--plan_only", "--dont_copy_preprocessed_data",
    ]
    check_call(cmd)

    # Export preprocessed dataset
    # nnU-Net will create a folder with the preprocessed data in the output directory.
    # Make sure you export any files that you want to save to the output directory!


if __name__ == '__main__':
    main()
