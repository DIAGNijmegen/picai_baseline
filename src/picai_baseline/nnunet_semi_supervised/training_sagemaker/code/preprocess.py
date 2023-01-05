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
    """Train nnU-Net semi-supervised model."""
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--workdir', type=str, default="/workdir")
    parser.add_argument('--imagesdir', type=str, default=os.environ.get('SM_CHANNEL_IMAGES', "/input/images"))
    parser.add_argument('--labelsdir', type=str, default=os.environ.get('SM_CHANNEL_LABELS', "/input/picai_labels"))
    parser.add_argument('--scriptsdir', type=str, default=os.environ.get('SM_CHANNEL_SCRIPTS', "/scripts"))
    parser.add_argument('--outputdir', type=str, default=os.environ.get('SM_MODEL_DIR', "/output"))
    parser.add_argument('--checkpointsdir', type=str, default="/checkpoints")
    parser.add_argument('--splits', type=str, default="picai_pubpriv",
                        help="Cross-validation splits. Can be a path to a json file or one of the predefined splits: "
                             "picai_pub, picai_pubpriv, picai_pub_nnunet, picai_pubpriv_nnunet, picai_debug.")
    parser.add_argument('--nnUNet_n_proc_DA', type=int, default=None)
    parser.add_argument('--nnUNet_tf', type=int, default=8, help="Number of preprocessing threads for full images")

    args, _ = parser.parse_known_args()

    # paths
    workdir = Path(args.workdir)
    images_dir = Path(args.imagesdir)
    labels_dir = Path(args.labelsdir)
    output_dir = Path(args.outputdir)
    scripts_dir = Path(args.scriptsdir)
    splits_path = workdir / f"splits/{taskname}/splits.json"

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_path.parent.mkdir(parents=True, exist_ok=True)

    # set environment variables
    os.environ["prepdir"] = str(workdir / "nnUNet_preprocessed")

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

    # # install modified nnU-Net
    # print("Installing modified nnU-Net...")
    # cmd = [
    #     "pip",
    #     "install",
    #     "-e",
    #     str(local_scripts_dir / "nnunet"),
    # ]
    # check_call(cmd)

    # resolve cross-validation splits
    predefined_splits = {
        "picai_pub": picai_pub_splits,
        "picai_pubpriv": picai_pubpriv_splits,
        "picai_pub_nnunet": picai_pub_nnunet_splits,
        "picai_pubpriv_nnunet": picai_pubpriv_nnunet_splits,
        "picai_debug": picai_debug_splits,
    }
    if args.splits in predefined_splits:
        nnunet_splits = predefined_splits[args.splits]
    else:
        # `splits` should be the path to a json file containing the splits
        print(f"Loading splits from {args.splits}")
        with open(args.splits, "r") as f:
            nnunet_splits = json.load(f)

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
        "--preprocessing_kwargs", '{"physical_size": [81.0, 192.0, 192.0], "crop_only": true}',
    ]
    check_call(cmd)

    # Preprocess data with nnU-Net
    print("Preprocessing data with nnU-Net...")
    cmd = [
        "nnunet", "plan_train", str(taskname), workdir.as_posix(),
        "--custom_split", str(splits_path),
        "--plan_only",
    ]
    check_call(cmd)

    # Export preprocessed dataset
    dst = output_dir / f"nnUNet_preprocessed/{taskname}/"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workdir / f"nnUNet_preprocessed/{taskname}/", dst)


if __name__ == '__main__':
    main()
