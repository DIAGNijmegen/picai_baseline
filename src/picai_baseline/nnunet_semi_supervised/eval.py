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

from picai_baseline.nnunet.eval import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command Line Arguments')
    parser.add_argument("--task", type=str, default="Task2203_picai_baseline")
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
                        help="Function to post-process the softmax volumes. Default: extract lesion candidates using the Report-Guided Annotation repository.")
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
