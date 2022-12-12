#!/opt/conda/bin/python
# Adapted from https://github.com/DIAGNijmegen/diag-nnunet/blob/master/nnunet_wrapper.py
import argparse
import functools
import json
import os
import pickle
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
from carbontracker.tracker import CarbonTracker
from nndet.io import get_training_dir
from scripts.consolidate import main as _consolidate
from scripts.predict import main as _predict

import shutil_sol

# add a flush to each print statement to ensure the stuff gets logged on SOL
print = functools.partial(print, flush=True)

Pathlike = Union[Path, str]


class CustomizedCarbonTracker:
    def __init__(self, logdir, enabled=True):
        if enabled:
            self.tracker = CarbonTracker(epochs=1, ignore_errors=True, devices_by_pid=False, log_dir=str(logdir), verbose=2)
        else:
            self.tracker = None

    def __enter__(self):
        if self.enabled:
            self.tracker.epoch_start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.tracker.epoch_end()
            self.tracker.stop()

    @property
    def enabled(self):
        return self.tracker is not None


def get_task_id(task_name):
    return re.match('Task([0-9]+)', task_name).group(1)


def print_split_per_fold(split_file, fold=None):
    try:
        with split_file.open('rb') as pkl:
            splits = pickle.load(pkl)
    except FileNotFoundError:
        print('Split file not found')
    else:
        for i, split in enumerate(splits):
            if fold not in (None, i):
                continue

            print(f'Fold #{i}')
            print('> Training')
            for caseid in sorted(split['train']):
                print(f'>> {caseid}')
            print('> Validation')
            for caseid in sorted(split['val']):
                print(f'>> {caseid}')

            if i + 1 < len(splits):
                print('-' * 25)


def copy_preprocessed_data_to_compute_node(task: str, remote_data_dir: Pathlike, custom_split=None):
    # resolve directories
    remote_task_dir = Path(remote_data_dir) / task
    local_task_dir = Path(os.environ['det_data']) / task
    splits_filepath = local_task_dir / 'preprocessed' / 'splits_final.pkl'

    if local_task_dir.absolute() == remote_task_dir.absolute():
        print(f"Remote and local task directory are the same: {local_task_dir}")
    else:
        # Copy preprocessed data to compute node
        print('[#] Copying plans and preprocessed data from storage server to compute node')
        local_task_dir.mkdir(parents=True, exist_ok=True)
        shutil_sol.copyfile(remote_task_dir / "dataset.json", local_task_dir / "dataset.json")
        shutil_sol.copytree(remote_task_dir / 'preprocessed', local_task_dir / 'preprocessed')

    # Replace split with custom split?
    if custom_split:
        splits = []
        with open(custom_split) as fp:
            for split in json.load(fp):
                splits.append(OrderedDict([
                    ('train', np.array(split['train'])),
                    ('val', np.array(split['val']))
                ]))

        splits_filepath.parent.mkdir(parents=True, exist_ok=True)
        with splits_filepath.open('wb') as fp:
            pickle.dump(splits, fp)
        print(f'[#] Replaced splits with custom split: {splits_filepath}')


def nndet_prep_train(argv):
    # Prepare experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('workdir', type=str, help='Path to working directory. Should contain the nnDet_raw_data folder.')
    parser.add_argument('--results', type=str, required=False)
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--custom_split', type=str, help='Path to a JSON file with a custom data split into five folds')
    parser.add_argument('--plan_only', action='store_true', help='Run the planning step, but not the training step')
    parser.add_argument('--validation_only', action='store_true', help='Do no run network training, only the final validation step')
    parser.add_argument('--carbontracker', action='store_true', help='Enables tracking of energy consumption')
    args = parser.parse_args(argv)

    # Set environment variables
    workdir = Path(args.workdir)
    taskid = get_task_id(args.task)
    remote_data_dir = workdir / 'nnDet_raw_data'
    remote_task_dir = remote_data_dir / args.task
    remote_task_prep_dir = remote_task_dir / 'preprocessed'
    local_data_dir = Path(os.environ['det_data'])
    local_task_dir = local_data_dir / args.task
    local_task_prep_dir = local_task_dir / 'preprocessed'
    splits_filepath = local_data_dir / args.task / 'preprocessed' / 'splits_final.pkl'
    os.environ['det_models'] = args.results if args.results else str(workdir / 'results/nnDet')

    # Start carbontracker
    with CustomizedCarbonTracker(local_data_dir / 'carbontracker', enabled=args.carbontracker):
        # Check if plans and preprocessed data are available
        if os.path.exists(remote_task_prep_dir / 'D3V001_3d.pkl'):
            if args.plan_only:
                print(f'[#] Found plans and preprocessed data for {args.task} - nothing to do')
        else:
            # Plans and data not available yet, run preprocessing on storage server
            os.environ['det_data'] = str(remote_data_dir)  # "trick" nnDetection to run on remote folder
            print('[#] Creating plans and preprocessing data')
            cmd = [
                'nndet_prep',
                taskid,
                # '--full_check'
            ]
            subprocess.check_call(cmd)

        # set data root to local data directory
        os.environ['det_data'] = str(local_data_dir)

        # copy preprocessed data to compute node
        copy_preprocessed_data_to_compute_node(
            task=args.task,
            remote_data_dir=remote_data_dir,
            custom_split=args.custom_split,
        )

        if args.plan_only:
            return

        print("[#] Unpacking data")
        path = local_task_prep_dir / "D3V001_3d/imagesTr"
        subprocess.check_call(["nndet_unpack", str(path), "6"])

        if not args.validation_only:
            print('[#] Starting network training')
            # Run training
            cmd = [
                "nndet_train",
                taskid,
                "-o", f"exp.fold={args.fold}"
            ]
            subprocess.check_call(cmd)
        else:
            raise ValueError("Validation only is not supported yet")

        # print('[#] Running validation')
        # cmd = [
        #     "nndet_eval",
        #     taskid,
        #     "RetinaUNetV001_D3V001_3d",
        #     args.fold,
        #     "--boxes",
        #     "--analyze_boxes"
        # ]
        # subprocess.check_call(cmd)

        # Copy split file since that is for sure available now (nnUNet_train has created
        # it the file did not exist already - unless training with "all", so still check)
        if splits_filepath.exists():
            shutil_sol.copyfile(splits_filepath, remote_task_prep_dir)


def prepare_argv_for_nndet(argv, workdir=None):
    """Prepare argv for nndet/scripts/predict.py -> main"""
    argv = list(argv)
    for param in [
        "--input",
        "--output",
        "--results",
        "--custom_split",
    ]:
        if param in argv:
            idx = argv.index(param)
            argv.pop(idx)
            argv.pop(idx)

    for param in [
        "--resume",
        "--skip_consolidate",
    ]:
        if param in argv:
            argv.remove(param)

    if workdir is not None and workdir in argv:
        argv.remove(workdir)

    return argv


def nndet_predict(argv):
    """
    Use trained network to run inference

    Internally, nnDetection expects to read from:
    /home/user/data/{taskid}/raw_splitted/imagesTs
    So, we copy the /input folder there.

    Internally, nnDetection will store predictions in:
    /output/models/{taskid}/RetinaUNetV001_D3V001_3d/fold{0,1,2,3,4} or consolidated/test_predictions

    After inference is done, predictions are moved to --data folder.
    """
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument('model', type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument('workdir', type=str, help='Path to working directory. Should contain the nnDet_raw_data folder.')
    parser.add_argument('-r', '--results', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, help="Path to raw input images")
    parser.add_argument('-o', '--output', type=str, help="Path to store predictions")
    parser.add_argument('-f', '--fold', type=int, required=False, default=-1,
                        help="fold to use for prediction. -1 uses the consolidated model")
    parser.add_argument('-c', '--resume', action='store_true',
                        help="Whether to allow resuming inference (in intermediate prediction folder)")
    parser.add_argument('--skip_consolidate', action='store_true',
                        help="Whether to copy plan.pkl to plan_inference.pkl")
    args, _ = parser.parse_known_args(argv)

    print(f"[#] Task: {args.task}")
    print(f"[#] Input: {args.input}")
    print(f"[#] Output: {args.output}")

    # set environment variable
    os.environ['det_models'] = args.results

    # check intermediate prediction directory
    task_model_dir = Path(os.environ['det_models'])
    training_dir = get_training_dir(task_model_dir / args.task / args.model, args.fold)
    initial_pred_dir = training_dir / "test_predictions"
    if initial_pred_dir.exists() and os.listdir(initial_pred_dir) and not args.resume:
        # predictions exist at temporary prediction directory, but resuming is not enabled
        raise ValueError("Intermediate prediction folder is not empty. Use --resume to allow resuming inference.")

    print("[#] Copying dataset.json and raw cases to local compute node for inference")
    # copy dataset.json
    local_task_dir = Path(os.environ['det_data']) / args.task
    local_task_dir.mkdir(parents=True, exist_ok=True)
    shutil_sol.copyfile(
        src=Path(args.workdir) / "nnDet_raw_data" / args.task / "dataset.json",
        dst=local_task_dir / "dataset.json"
    )

    # copy raw cases
    local_dir_input = Path(os.environ['det_data']) / args.task / "raw_splitted/imagesTs"
    shutil_sol.copytree(args.input, local_dir_input)

    if args.skip_consolidate:
        print("[#] Copying plan.pkl to plan_inference.pkl. This is not removed afterwards! I think this uses model_best + model_last.")
        shutil_sol.copyfile(training_dir / "plan.pkl", training_dir / "plan_inference.pkl")

    # run prediction script with remaining arguments (including taskid)
    argv = prepare_argv_for_nndet(argv, workdir=args.workdir)
    sys.argv = [sys.argv[0]] + argv
    _predict()

    if not os.path.exists(args.output):
        # move predictions to output folder
        print("[#] Renaming intermediate prediction folder to output folder")
        os.rename(initial_pred_dir, args.output)
    else:
        # move predictions to output folder
        print("[#] Merging intermediate prediction folder with output folder (existing files are overwritten)")
        for fn in os.listdir(initial_pred_dir):
            os.rename(initial_pred_dir / fn, Path(args.output) / fn)

    print("[#] Finished.")


def nndet_consolidate(argv):
    """Consolidate trained nnDetection models to consolidated folder for ensembling"""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument('model', type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument('workdir', type=str, help='Path to working directory. Should contain the nnDet_raw_data folder.')
    parser.add_argument('-r', '--results', type=str, required=True)
    parser.add_argument('--custom_split', type=str, help='Path to a JSON file with a custom data split into five folds')
    args, _ = parser.parse_known_args(argv)

    print(f"[#] Task: {args.task}")
    print(f"[#] Model: {args.model}")
    print(f"[#] Results: {args.results}")

    # set environment variable
    os.environ['det_models'] = args.results

    # we require preprocessed data for nndet sweep/consolitate
    if not os.path.exists(Path(os.environ['det_data']) / args.task / "dataset.json"):
        copy_preprocessed_data_to_compute_node(
            task=args.task,
            remote_data_dir=Path(args.workdir) / "nnDet_raw_data",
            custom_split=args.custom_split,
        )

    # check if sweep for hyperparameter tuning must be ran
    for fold in range(5):
        sweep_results = Path(args.results, args.task, args.model, f'fold{fold}/sweep/sweep_remove_small_boxes.json')
        if os.path.exists(sweep_results):
            continue

        # run parameter sweep
        print(f"[#] Running parameter sweep for fold {fold}")
        cmd = [
            "nndet_sweep",
            args.task,
            args.model,
            str(fold),
        ]
        subprocess.check_call(cmd)

    # run prediction script with remaining arguments (including taskid)
    argv = prepare_argv_for_nndet(argv, workdir=args.workdir)
    sys.argv = [sys.argv[0]] + argv
    _consolidate()

    print("[#] Finished.")


if __name__ == '__main__':
    # Very first argument determines action
    # Full process: nndet_prep -> nndet_unpack -> nndet_train -> nndet_consolidate -> nndet_predict
    actions = {
        'prep_train': nndet_prep_train,
        # 'unpack': nndet_unpack, # Copy to SSD then unpack, ToDo: incorporate into training?
        # 'train': nndet_train,
        'consolidate': nndet_consolidate,  # Finds best hyperparameters for inference, ToDo: incorporate into predict?
        'predict': nndet_predict,
        # 'evaluate': nndet_eval # --test for use with self-made test set
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: nndet ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])
