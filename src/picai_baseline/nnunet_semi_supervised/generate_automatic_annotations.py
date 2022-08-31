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
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

nnUNet_results = Path("/workdir/results/nnUNet/3d_fullres/")
nnUNet_task = nnUNet_results / "Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1"
softmax_dir = nnUNet_task / "picai_pubtrain_predictions_model_best_ensemble"
autumatic_annotation_dir = nnUNet_task / "picai_pubtrain_predictions_model_best_ensemble_automatic_annotations"
nnunet_raw_data_path = Path("/repos/picai_labels/clinical_information/marksheet.csv")
num_lesions_to_retain_map_path = softmax_dir / "num_lesions_to_retain_map.json"

df = pd.read_csv(nnunet_raw_data_path)

num_lesions_to_retain_map = {}


@dataclass
class Gleason_score(object):
    """Gleason score with pattern1 + pattern2 = total"""
    pattern1: int
    pattern2: int

    def __post_init__(self):
        assert isinstance(self.pattern1, int) and self.pattern1 >= 0, f"Got invalid pattern1: {self.pattern1}"
        assert isinstance(self.pattern2, int) and self.pattern2 >= 0, f"Got invalid pattern2: {self.pattern2}"

    @property
    def total(self):
        return self.pattern1 + self.pattern2

    @property
    def GGG(self):
        if self.pattern1 == 0 and self.pattern2 == 0:
            return 0
        elif self.pattern1 + self.pattern2 >= 9:
            return 5
        elif self.pattern1 + self.pattern2 == 8:
            return 4
        elif self.pattern1 + self.pattern2 == 7:
            return 3 if self.pattern1 == 4 else 2
        elif self.pattern1 + self.pattern2 <= 6:
            return 1

    def __repr__(self):
        return f"Gleason score({self.pattern1}+{self.pattern2}={self.total})"

    def __str__(self):
        return self.__repr__()

    def to_tuple(self):
        return self.pattern1, self.pattern2, self.total

for i, row in df.iterrows():
    # grab lesion Gleason scores scores
    if isinstance(row.lesion_GS, float) and np.isnan(row.lesion_GS):
        gleason_scores = []
    else:
        gleason_scores = row.lesion_GS.split(",")

    # convert Gleason scores to ISUP grades
    isup_grades = []
    for score in gleason_scores:
        if score == "N/A":
            continue

        pattern1, pattern2 = score.split("+")
        GS = Gleason_score(int(pattern1), int(pattern2))
        isup_grades.append(GS.GGG)

    fn = f"{row.patient_id}_{row.study_id}.nii.gz"
    num_lesions_to_retain_map[fn] = sum([score >= 2 for score in isup_grades])

# write to JSON
with open(num_lesions_to_retain_map_path, "w") as fp:
    json.dump(num_lesions_to_retain_map, fp, indent=4)


# generate automatic annotations
autumatic_annotation_dir.mkdir(exist_ok=True)
cmd = [
    "python", "-m", "report_guided_annotation",
    "--input", str(softmax_dir),
    "--output", str(autumatic_annotation_dir),
    "--threshold", "dynamic",
    "--skip_if_insufficient_lesions", "0"
]
subprocess.check_call(cmd)
