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
from picai_prep.examples.mha2nnunet.picai_archive import \
    generate_mha2nnunet_settings

from picai_baseline.splits.picai_nnunet import nnunet_splits

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
task = "Task2201_picai_baseline"

# paths
mha_archive_dir = inputdir / "images"
annotations_dir = inputdir / "labels/csPCa_lesion_delineations/human_expert/resampled/"
mha2nnunet_settings_path = workdir / "mha2nnunet_settings" / "Task2201_picai_baseline.json"
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


if mha2nnunet_settings_path.exists():
    print(f"Found mha2nnunet settings at {mha2nnunet_settings_path}, skipping..")
else:
    # generate mha2nnunet conversion plan
    Path(mha2nnunet_settings_path.parent).mkdir(parents=True, exist_ok=True)
    generate_mha2nnunet_settings(
        archive_dir=mha_archive_dir,
        annotations_dir=annotations_dir,
        output_path=mha2nnunet_settings_path,
        task=task
    )

    # read mha2nnunet_settings
    with open(mha2nnunet_settings_path) as fp:
        mha2nnunet_settings = json.load(fp)

    # note: modify preprocessing settings here
    mha2nnunet_settings["preprocessing"]["physical_size"] = [81.0, 192.0, 192.0]
    mha2nnunet_settings["preprocessing"]["crop_only"] = True

    # save mha2nnunet_settings
    with open(mha2nnunet_settings_path, "w") as fp:
        json.dump(mha2nnunet_settings, fp, indent=4)
    print(f"Saved updated mha2nnunet settings to {mha2nnunet_settings_path}")


if nnUNet_dataset_json_path.exists():
    print(f"Found dataset.json at {nnUNet_dataset_json_path}, skipping..")
else:
    # prepare dataset in nnUNet format
    archive = MHA2nnUNetConverter(
        input_path=mha_archive_dir,
        annotations_path=annotations_dir,
        output_path=nnUNet_raw_data_path,
        settings_path=mha2nnunet_settings_path,
        lbl_preprocess_func=preprocess_picai_annotation,
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
