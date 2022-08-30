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

from pathlib import Path

from picai_baseline.splits.picai_nnunet import valid_splits
from picai_eval import evaluate_folder
from picai_prep.preprocessing import crop_or_pad
from report_guided_annotation import extract_lesion_candidates


def nnUNet_cropped_prediction_compatibility(img):
    return crop_or_pad(img, size=(20, 320, 320))


def extract_lesion_candidates_compatibility(pred):
    return nnUNet_cropped_prediction_compatibility(extract_lesion_candidates(pred)[0])


task = "Task2203_picai_baseline"
trainer = "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1"
results_dir = Path("/workdir/results/nnUNet/3d_fullres/")

for fold in range(5):
    print(f"Fold {fold}")
    if fold == 0:
        checkpoints = ["model_best", "model_final_checkpoint"] + [
            f"model_ep_{epoch:03d}" for epoch in range(50, 950+1, 50)
        ]
    else:
        checkpoints = ["model_best"]

    for checkpoint in checkpoints:
        softmax_dir = results_dir / task / trainer / f"fold_{fold}/picai_pubtrain_predictions_{checkpoint}"
        metrics_path = softmax_dir.parent  / f"metrics-{checkpoint}.json"

        if metrics_path.exists():
            print(f"Metrics found at {metrics_path}, skipping..")
            continue

        try:
            # evaluate
            metrics = evaluate_folder(
                y_det_dir=softmax_dir,
                y_true_dir=f"/workdir/nnUNet_raw_data/{task}/labelsTr",
                subject_list=valid_splits[fold]['subject_list'],
                y_det_postprocess_func=extract_lesion_candidates_compatibility,
                y_true_postprocess_func=nnUNet_cropped_prediction_compatibility,
            )

            # save and show metrics
            metrics.save(metrics_path)
            print(f"Results for checkpoint {checkpoint}:")
            print(metrics)
        except Exception as e:
            print(f"Error for checkpoint {checkpoint}: {e}")
