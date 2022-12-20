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
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import SimpleITK as sitk

from picai_baseline.splits.picai import nnunet_splits as picai_pub_splits
from picai_baseline.splits.picai_nnunet import \
    nnunet_splits as picai_pub_nnunet_splits
from picai_baseline.splits.picai_pubpriv import \
    nnunet_splits as picai_pubpriv_splits
from picai_baseline.splits.picai_pubpriv_nnunet import \
    nnunet_splits as picai_pubpriv_nnunet_splits


def main(
    preprocessed_data_path: Union[Path, str] = Path('/workdir/nnUNet_raw_data/Task2201_picai_baseline/'),
    overviews_path: Union[Path, str] = Path('/workdir/results/UNet/overviews/'),
    splits: Optional[Dict[str, List[str]]] = None,
):
    """Create overviews of the training data."""

    if isinstance(splits, str):
        # select splits
        splits = {
            "picai_pub": picai_pub_splits,
            "picai_pubpriv": picai_pubpriv_splits,
            "picai_pub_nnunet": picai_pub_nnunet_splits,
            "picai_pubpriv_nnunet": picai_pubpriv_nnunet_splits,
        }[args.splits]

    if splits is None:
        raise ValueError("No splits provided!")

    # create directory to store overviews
    overviews_path.mkdir(parents=True, exist_ok=True)

    # iterate over each cross-validation fold
    for fold, nnunet_fold in enumerate(splits):

        # iterate over train and validation splits
        for split, nnunet_split in nnunet_fold.items():
            print(f"Preparing fold {fold} ({split})..")

            # initialize list of fields to collect for each split of each fold
            overview = {
                'pat_ids': [],
                'study_ids': [],
                'image_paths': [],
                'label_paths': [],
                'case_label': [],
                'ratio_csPCa_bg': []
            }

            # iterate over each training/validation case
            for subject_id in nnunet_split:
                patient_id, study_id = subject_id.split('_')

                # skip cases where preprocessing failed
                if not (preprocessed_data_path / 'labelsTr' / f'{subject_id}.nii.gz').exists():
                    print(f"Skipping {subject_id}, case not found (likely because preprocessing failed)!")
                    continue

                # load annotation
                lbl = sitk.GetArrayFromImage(sitk.ReadImage(str(preprocessed_data_path / 'labelsTr' / f'{subject_id}.nii.gz')))

                overview['pat_ids'] += [patient_id]
                overview['study_ids'] += [study_id]
                overview['image_paths'] += [[
                    str((preprocessed_data_path / 'imagesTr' / f'{subject_id}_0000.nii.gz').as_posix()),
                    str((preprocessed_data_path / 'imagesTr' / f'{subject_id}_0001.nii.gz').as_posix()),
                    str((preprocessed_data_path / 'imagesTr' / f'{subject_id}_0002.nii.gz').as_posix()),
                ]]
                overview['label_paths'] += [str((preprocessed_data_path / 'labelsTr' / f'{subject_id}.nii.gz').as_posix())]
                overview['case_label'] += [float(np.max(lbl))]
                overview['ratio_csPCa_bg'] += [float(np.sum(lbl)/np.size(lbl))]

            # save overview
            with open(overviews_path / f'PI-CAI_{split}-fold-{fold}.json', 'w') as fp:
                json.dump(overview, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command Line Arguments')
    parser.add_argument("--task", type=str, default="Task2201_picai_baseline",
                        help="Task name of the experiment. Default: Task2201_picai_baseline")
    parser.add_argument("--workdir", type=str, default="/workdir",
                        help="Path to the workdir where 'results' and 'nnUNet_raw_data' are stored. Default: /workdir")
    parser.add_argument("--preprocessed_data_path", type=str, default="nnUNet_raw_data/{task}",
                        help="Path to the preprocessed data, relative to the workdir. Default: /workdir/nnUNet_raw_data/{task}")
    parser.add_argument("--overviews_path", type=str, default="results/UNet/overviews/{task}",
                        help="Path to the overviews, relative to the workdir. Default: /workdir/results/UNet/overviews/{task}")
    parser.add_argument("--splits", type=str, default="picai_pub_nnunet",
                        help="Splits for cross-validation. Available: picai_pub, picai_pub_nnunet, picai_pubpriv, " +
                             "picai_pubpriv_nnunet.")
    args = parser.parse_args()

    # paths
    workdir = Path(args.workdir)
    preprocessed_data_path = workdir / args.preprocessed_data_path.replace("{task}", args.task)
    overviews_path = workdir / args.overviews_path.replace("{task}", args.task)

    # evaluate
    main(
        preprocessed_data_path=preprocessed_data_path,
        overviews_path=overviews_path,
        splits=args.splits,
    )
    print("Finished.")
