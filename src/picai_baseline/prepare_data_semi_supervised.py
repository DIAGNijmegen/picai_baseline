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
import os
from pathlib import Path

import SimpleITK as sitk
from picai_prep import MHA2nnUNetConverter
from picai_prep.data_utils import atomic_image_write
from picai_prep.examples.mha2nnunet.picai_archive import \
    generate_mha2nnunet_settings
from tqdm import tqdm

from picai_baseline.splits.picai import nnunet_splits

"""
Script to prepare PI-CAI data into the nnUNet raw data format
For documentation, please see:
https://github.com/DIAGNijmegen/picai_baseline#prepare-data
"""

# environment settings
if 'inputdir' in os.environ:
    inputdir = Path(os.environ['inputdir'])
else:
    inputdir = Path("/input")
if 'workdir' in os.environ:
    workdir = Path(os.environ['workdir'])
else:
    workdir = Path("/workdir")

# settings
task = "Task2203_picai_baseline"

# paths
mha_archive_dir = inputdir / "images"
annotations_dir_human_expert = inputdir / "picai_labels/csPCa_lesion_delineations/human_expert/resampled/"
annotations_dir_ai_derived = inputdir / "picai_labels/csPCa_lesion_delineations/AI/Bosma22a/"
annotations_dir = inputdir / "picai_labels/csPCa_lesion_delineations/combined/"
mha2nnunet_settings_path = workdir / "mha2nnunet_settings" / (task + ".json")
nnUNet_raw_data_path = workdir / "nnUNet_raw_data"
nnUNet_task_dir = nnUNet_raw_data_path / task
nnUNet_dataset_json_path = nnUNet_task_dir / "dataset.json"
nnUNet_splits_path = nnUNet_task_dir / "splits.json"


def preprocess_picai_annotation(lbl: sitk.Image) -> sitk.Image:
    """Binarize the granular ISUP ≥ 2 annotations"""
    lbl_arr = sitk.GetArrayFromImage(lbl)

    # convert granular PI-CAI csPCa annotation to binary csPCa annotation
    lbl_arr = (lbl_arr >= 1).astype('uint8')

    # convert label back to SimpleITK
    lbl_new: sitk.Image = sitk.GetImageFromArray(lbl_arr)
    lbl_new.CopyInformation(lbl)
    return lbl_new


# prepare annotations folder with human-expert and AI-derived annotations
annotations_dir.mkdir(parents=True, exist_ok=True)
if len(os.listdir(annotations_dir)) == 1500:
    print("Annotations folder already prepared, skipping..")
else:
    print("Preparing annotations folder with human-expert and AI-derived annotations")
    for fn in tqdm(os.listdir(annotations_dir_human_expert), desc="Copying human expert annotations"):
        path_src = annotations_dir_human_expert / fn
        path_dst = annotations_dir / fn
        if fn.endswith(".nii.gz") and not path_dst.exists():
            lbl = sitk.ReadImage(str(path_src))
            atomic_image_write(lbl, path_dst)
    for fn in tqdm(os.listdir(annotations_dir_ai_derived), desc="Copying AI-derived annotations"):
        path_src = annotations_dir_ai_derived / fn
        path_dst = annotations_dir / fn
        if fn.endswith(".nii.gz") and not path_dst.exists():
            lbl = sitk.ReadImage(str(path_src))
            atomic_image_write(lbl, path_dst)

if mha2nnunet_settings_path.exists():
    print(f"Found mha2nnunet settings at {mha2nnunet_settings_path}, skipping..")
else:
    # generate mha2nnunet conversion plan
    Path(mha2nnunet_settings_path.parent).mkdir(parents=True, exist_ok=True)
    generate_mha2nnunet_settings(
        archive_dir=mha_archive_dir,
        annotations_dir=annotations_dir,
        output_path=mha2nnunet_settings_path,
    )

    # read mha2nnunet_settings
    with open(mha2nnunet_settings_path) as fp:
        mha2nnunet_settings = json.load(fp)

    # note: modify preprocessing settings here
    mha2nnunet_settings["dataset_json"]["task"] = task
    # mha2nnunet_settings["preprocessing"]["matrix_size"] = [20, 256, 256]
    # mha2nnunet_settings["preprocessing"]["spacing"] = [3.0, 0.5, 0.5]

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
        scans_dir=mha_archive_dir,
        annotations_dir=annotations_dir,
        mha2nnunet_settings=mha2nnunet_settings,
    )
    archive.convert()
    archive.generate_json()

if nnUNet_splits_path.exists():
    print(f"Found cross-validation splits at {nnUNet_splits_path}, skipping..")
else:
    # save cross-validation splits to disk
    with open(nnUNet_splits_path, "w") as fp:
        json.dump(nnunet_splits, fp)
    print(f"Saved cross-validation splits to {nnUNet_splits_path}")

print("Finished.")
