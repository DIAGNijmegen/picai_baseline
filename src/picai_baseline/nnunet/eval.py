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
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from picai_baseline.nnunet.softmax_export import \
    convert_cropped_npz_to_original_nifty
from picai_baseline.splits.picai_nnunet import valid_splits
from picai_eval import evaluate_folder
from picai_prep.preprocessing import crop_or_pad
from report_guided_annotation import extract_lesion_candidates

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass


def extract_lesion_candidates_cropped(pred, threshold):
    size = pred.shape
    pred = crop_or_pad(pred, (20, 384, 384))
    pred = crop_or_pad(pred, size)
    return extract_lesion_candidates(pred, threshold=threshold)[0]


def evaluate(
    task: str = "Task2201_picai_baseline",
    trainer: str = "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
    workdir: Union[Path, str] = "/workdir",
    task_dir: Union[Path, str] = "auto",
    checkpoints: str = ["model_best"],
    folds: str = list(range(5)),
    softmax_postprocessing_func: "Optional[Union[Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]], str]]" = "extract_lesion_candidates",
    threshold: str = "dynamic",
    metrics_fn: str = "metrics-{checkpoint}-{threshold}.json",
):
    # input validation
    workdir = Path(workdir)
    if task_dir == "auto":
        task_dir = workdir / "results" / "nnUNet" / "3d_fullres" / task
    else:
        task_dir = workdir / task_dir

    if isinstance(softmax_postprocessing_func, str):
        if softmax_postprocessing_func == "extract_lesion_candidates":
            softmax_postprocessing_func = lambda pred: extract_lesion_candidates(pred, threshold=threshold)[0]
        elif softmax_postprocessing_func == "extract_lesion_candidates_cropped":
            softmax_postprocessing_func = lambda pred: extract_lesion_candidates_cropped(pred, threshold=threshold)
        else:
            raise ValueError(f"Unrecognised softmax_postprocessing_func: {softmax_postprocessing_func}")

    for fold in folds:
        print(f"Evaluating fold {fold}...")

        for checkpoint in checkpoints:
            softmax_dir = task_dir / f"{trainer}__nnUNetPlansv2.1" / f"fold_{fold}/picai_pubtrain_predictions_{checkpoint}"
            metrics_fn = metrics_fn.replace(r"{checkpoint}", checkpoint).replace(r"{threshold}", threshold)
            metrics_path = softmax_dir.parent / metrics_fn

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
                y_true_dir=workdir / "nnUNet_raw_data" / task / "labelsTr",
                subject_list=valid_splits[fold]['subject_list'],
                pred_extensions=['_softmax.nii.gz'],
                y_det_postprocess_func=softmax_postprocessing_func,
                num_parallel_calls=5,
            )

            # save and show metrics
            metrics.save(metrics_path)
            print(f"Results for checkpoint {checkpoint}:")
            print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command Line Arguments')
    parser.add_argument("--task", type=str, default="Task2201_picai_baseline",
                        help="Task name of the nnU-Net experiment. Default: Task2201_picai_baseline")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
                        help="Trainer of the nnU-Net experiment. Default: nnUNetTrainerV2_Loss_FL_and_CE_checkpoints")
    parser.add_argument("--workdir", type=str, default="/workdir",
                        help="Path to the workdir where 'results' and 'nnUNet_raw_data' are stored. Default: /workdir")
    parser.add_argument("--task_dir", type=str, default="auto",
                        help="Path to the task directory (relative to the workdir). Optional, will be constucted for default nnU-Net forlder structure")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=["model_best"],
                        help="Which checkpoints to evaluate. Multiple checkpoints can be passed at once. Default: model_best")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)),
                        help="Which folds to evaluate. Multiple folds can be evaluated at once. Default: 0, 1, 2, 3, 4  (all)")
    parser.add_argument("--softmax_postprocessing_func", type=str, default="extract_lesion_candidates",
                        help="Function to post-process the softmax volumes. Default: extract lesion candidates using the Report-Guided Annotation repository. " + \
                             "Use extract_lesion_candidates_cropped to set all predictions outside the central 20x384x384 voxels to zero.")
    parser.add_argument("--threshold", type=str, default="dynamic",
                        help="Threshold for lesion extraction from softmax predictions. " + \
                             "Use dynamic-fast for quicker evaluation at almost equal performance.")
    parser.add_argument("--metrics_fn", type=str, default=r"metrics-{checkpoint}-{threshold}.json",
                        help=r"Filename to save metrics to. May contain {checkpoint} and {threshold} which are auto-filled. Default: metrics-{checkpoint}-{threshold}.json")
    args = parser.parse_args()

    # evaluate
    evaluate(
        task=args.task,
        trainer=args.trainer,
        workdir=args.workdir,
        task_dir=args.task_dir,
        checkpoints=args.checkpoints,
        folds=args.folds,
        softmax_postprocessing_func=args.softmax_postprocessing_func,
        threshold=args.threshold,
        metrics_fn=args.metrics_fn,
    )
