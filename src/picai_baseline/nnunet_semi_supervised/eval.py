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

from multiprocessing import Pool
from pathlib import Path

from picai_baseline.nnunet.softmax_export import \
    convert_cropped_npz_to_original_nifty
from picai_baseline.splits.picai_nnunet import valid_splits
from picai_eval import evaluate_folder
from report_guided_annotation import extract_lesion_candidates

task = "Task2203_picai_baseline"
trainer = "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1"
task_dir = Path("/workdir/results/nnUNet/3d_fullres/") / task
checkpoints = ["model_best"]
threshold = "dynamic"  # use dynamic-fast for quicker evaluation at almost equal performance

for fold in range(5):
    print(f"Fold {fold}")

    for checkpoint in checkpoints:
        softmax_dir = task_dir / trainer / f"fold_{fold}/picai_pubtrain_predictions_{checkpoint}"
        metrics_path = softmax_dir.parent  / f"metrics-{checkpoint}-picai_eval-{threshold}.json"

        if metrics_path.exists():
            print(f"Metrics found at {metrics_path}, skipping..")
            continue

        # pad raw npz predictions to their original extent
        with Pool() as pool:
            pool.map(
                func=convert_cropped_npz_to_original_nifty,
                iterable=softmax_dir.glob("*.npz"),
            )

        # evaluate
        metrics = evaluate_folder(
            y_det_dir=softmax_dir,
            y_true_dir=f"/workdir/nnUNet_raw_data/{task}/labelsTr",
            subject_list=valid_splits[fold]['subject_list'],
            pred_extensions=['_softmax.nii.gz'],
            y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred, threshold=threshold)[0],
        )

        # save and show metrics
        metrics.save(metrics_path)
        print(f"Results for checkpoint {checkpoint}:")
        print(metrics)
