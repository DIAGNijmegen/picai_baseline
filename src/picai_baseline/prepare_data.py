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
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import SimpleITK as sitk
from picai_prep import MHA2nnUNetConverter
from picai_prep.examples.mha2nnunet.picai_archive import \
    generate_mha2nnunet_settings

from picai_baseline.splits.picai import nnunet_splits as picai_pub_splits
from picai_baseline.splits.picai_debug import \
    nnunet_splits as picai_debug_splits
from picai_baseline.splits.picai_nnunet import \
    nnunet_splits as picai_pub_nnunet_splits
from picai_baseline.splits.picai_pubpriv import \
    nnunet_splits as picai_pubpriv_splits
from picai_baseline.splits.picai_pubpriv_nnunet import \
    nnunet_splits as picai_pubpriv_nnunet_splits

"""
Script to prepare PI-CAI data into the nnUNet raw data format
For documentation, please see:
https://github.com/DIAGNijmegen/picai_baseline#prepare-data
"""


def preprocess_picai_annotation(lbl: sitk.Image) -> sitk.Image:
    """Binarize the granular ISUP ≥ 2 annotations"""
    lbl_arr = sitk.GetArrayFromImage(lbl)

    # convert granular PI-CAI csPCa annotation to binary csPCa annotation
    lbl_arr = (lbl_arr >= 1).astype('uint8')

    # convert label back to SimpleITK
    lbl_new: sitk.Image = sitk.GetImageFromArray(lbl_arr)
    lbl_new.CopyInformation(lbl)
    return lbl_new


def prepare_data(
    workdir: Union[Path, str] = "/workdir",
    inputdir: Union[Path, str] = "/input",
    imagesdir: str = "images",
    labelsdir: str = "picai_labels",
    spacing: Optional[Iterable[float]] = None,
    matrix_size: Optional[Iterable[int]] = None,
    preprocessing_kwargs: Optional[Dict[str, Any]] = None,
    splits: str = "picai_pub_nnunet",
    task: str = "Task2201_picai_baseline",
):

    # prepare preprocessing kwargs
    if preprocessing_kwargs is None or preprocessing_kwargs == "":
        preprocessing_kwargs = {}
    elif isinstance(preprocessing_kwargs, str):
        preprocessing_kwargs = json.loads(preprocessing_kwargs)
    if not isinstance(preprocessing_kwargs, dict):
        raise ValueError("preprocessing_kwargs must be a dict or None")
    if spacing:
        if "spacing" in preprocessing_kwargs:
            raise ValueError("Cannot specify both --spacing and --preprocessing_kwargs['spacing']")
        preprocessing_kwargs["spacing"] = spacing
    if matrix_size:
        if "matrix_size" in preprocessing_kwargs:
            raise ValueError("Cannot specify both --matrix_size and --preprocessing_kwargs['matrix_size']")
        preprocessing_kwargs["matrix_size"] = matrix_size

    # resolve cross-validation splits
    predefined_splits = {
        "picai_pub": picai_pub_splits,
        "picai_pubpriv": picai_pubpriv_splits,
        "picai_pub_nnunet": picai_pub_nnunet_splits,
        "picai_pubpriv_nnunet": picai_pubpriv_nnunet_splits,
        "picai_debug": picai_debug_splits,
    }
    if isinstance(splits, str) and splits in predefined_splits:
        splits = predefined_splits[splits]
    else:
        # `splits` should be the path to a json file containing the splits
        print(f"Loading splits from {splits}")
        with open(splits, "r") as f:
            splits = json.load(f)

    # parse paths
    workdir = Path(workdir)
    inputdir = Path(inputdir)
    imagesdir = Path(inputdir / imagesdir)
    labelsdir = Path(inputdir / labelsdir)

    # paths
    annotations_dir = labelsdir / "csPCa_lesion_delineations/human_expert/resampled/"
    mha2nnunet_settings_path = workdir / "mha2nnunet_settings" / f"{task}.json"
    nnUNet_raw_data_path = workdir / "nnUNet_raw_data"
    nnUNet_task_dir = nnUNet_raw_data_path / task
    nnUNet_dataset_json_path = nnUNet_task_dir / "dataset.json"
    nnUNet_splits_path = nnUNet_task_dir / "splits.json"

    if mha2nnunet_settings_path.exists():
        print(f"Found mha2nnunet settings at {mha2nnunet_settings_path}, skipping..")
    else:
        # generate mha2nnunet conversion plan
        Path(mha2nnunet_settings_path.parent).mkdir(parents=True, exist_ok=True)
        generate_mha2nnunet_settings(
            archive_dir=imagesdir,
            annotations_dir=annotations_dir,
            output_path=mha2nnunet_settings_path,
            task=task,
        )

        # read mha2nnunet_settings
        with open(mha2nnunet_settings_path) as fp:
            mha2nnunet_settings = json.load(fp)

        # note: modify preprocessing settings here
        mha2nnunet_settings["preprocessing"].update(preprocessing_kwargs)

        # save mha2nnunet_settings
        with open(mha2nnunet_settings_path, "w") as fp:
            json.dump(mha2nnunet_settings, fp, indent=4)
        print(f"Saved mha2nnunet settings to {mha2nnunet_settings_path}")


    if nnUNet_dataset_json_path.exists():
        print(f"Found dataset.json at {nnUNet_dataset_json_path}, skipping..")
    else:
        # read preprocessing settings and set the annotation preprocessing function
        with open(mha2nnunet_settings_path) as fp:
            mha2nnunet_settings = json.load(fp)

        if not "options" in mha2nnunet_settings:
            mha2nnunet_settings["options"] = {}
        mha2nnunet_settings["options"]["annotation_preprocess_func"] = preprocess_picai_annotation

        # prepare dataset in nnUNet format
        archive = MHA2nnUNetConverter(
            output_dir=nnUNet_raw_data_path,
            scans_dir=imagesdir,
            annotations_dir=annotations_dir,
            mha2nnunet_settings=mha2nnunet_settings,
        )
        archive.convert()
        archive.create_dataset_json()

    if nnUNet_splits_path.exists():
        print(f"Found cross-validation splits at {nnUNet_splits_path}, skipping..")
    else:
        # save cross-validation splits to disk
        with open(nnUNet_splits_path, "w") as fp:
            json.dump(splits, fp)
        print(f"Saved cross-validation splits to {nnUNet_splits_path}")


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default=os.environ.get("workdir", "/workdir"),
                        help="Path to the working directory (default: /workdir, or the environment variable 'workdir')")
    parser.add_argument("--inputdir", type=str, default=os.environ.get("inputdir", "/input"),
                        help="Path to the input dataset (default: /input, or the environment variable 'inputdir')")
    parser.add_argument("--imagesdir", type=str, default="images",
                        help="Path to the images, relative to --inputdir (default: /input/images)")
    parser.add_argument("--labelsdir", type=str, default="picai_labels",
                        help="Path to the labels, relative to --inputdir (root of picai_labels) (default: /input/picai_labels)")
    parser.add_argument("--spacing", type=float, nargs="+", required=False,
                        help="Spacing to preprocess images to. Default: keep as-is.")
    parser.add_argument("--matrix_size", type=int, nargs="+", required=False,
                        help="Matrix size to preprocess images to. Default: keep as-is.")
    parser.add_argument("--preprocessing_kwargs", type=str, required=False,
                        help='Preprocessing kwargs to pass to the MHA2nnUNetConverter. " + \
                            "E.g.: `{"crop_only": true}`. Must be valid json.')
    parser.add_argument("--splits", type=str, default="picai_pub_nnunet",
                        help="Splits to save for cross-validation. Available: picai_pub_nnunet, picai_pubpriv_nnunet.")
    parser.add_argument("--task", type=str, default="Task2201_picai_baseline",
                        help="Task name (default: Task2201_picai_baseline)")
    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"Parsing all arguments failed: {e}")
        print("Retrying with only the known arguments...")
        args, _ = parser.parse_known_args()

    prepare_data(
        workdir=args.workdir,
        inputdir=args.inputdir,
        imagesdir=args.imagesdir,
        labelsdir=args.labelsdir,
        spacing=args.spacing,
        matrix_size=args.matrix_size,
        preprocessing_kwargs=args.preprocessing_kwargs,
        splits=args.splits,
        task=args.task,
    )
    print("Finished.")
