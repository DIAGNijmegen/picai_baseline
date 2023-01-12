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
import os

from picai_baseline.prepare_data import prepare_data
from picai_baseline.prepare_data_semi_supervised import \
    prepare_data_semi_supervised

if __name__ == '__main__':
    # Set up command line arguments
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers()

    # parse command line arguments
    parser = subparsers.add_parser('prepare_data_semi_supervised')
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
    parser.add_argument("--splits", type=str, default="picai_pub",
                        help="Splits to save for cross-validation. Available: picai_pub, picai_pubpriv.")
    parser.add_argument("--task", type=str, default="Task2203_picai_baseline",
                        help="Task name (default: Task2203_picai_baseline)")
    parser.set_defaults(func=prepare_data_semi_supervised)

    parser = subparsers.add_parser('prepare_data')
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
    parser.set_defaults(func=prepare_data)

    try:
        args = argparser.parse_args()
    except Exception as e:
        print(f"Parsing all arguments failed: {e}")
        print("Retrying with only the known arguments...")
        args, _ = argparser.parse_known_args()

    args.func(
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
