#!/opt/conda/bin/python

import argparse
import functools
import os
import pickle
import re
import subprocess
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
from carbontracker.tracker import CarbonTracker
from nnunet.utilities import shutil_sol
from nnunet.utilities.io import (checksum, path_exists, read_json,
                                 refresh_file_list, write_json)
from picai_prep.data_utils import atomic_file_copy

PLANS = 'nnUNetPlansv2.1'

# add a flush to each print statement to ensure the stuff gets logged on SOL
print = functools.partial(print, flush=True)


class CustomizedCarbonTracker:
    def __init__(self, logdir, enabled=True):
        if enabled:
            self.tracker = CarbonTracker(epochs=1, ignore_errors=True, devices_by_pid=False, log_dir=str(logdir),
                                         verbose=2)
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


def prepare(argv):
    # Convert images and masks into expected format
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--masks', type=str, required=True)
    parser.add_argument('--modality', type=str, default='CT')
    parser.add_argument('--labels', type=str, nargs='*', default=['background', 'foreground'])
    parser.add_argument('--license', type=str, default='')
    parser.add_argument('--release', type=str, default='1.0')
    args = parser.parse_args(argv)

    # Check if task already exists
    print('[#] Creating directory structure')

    datadir = Path(args.data)
    taskdir = datadir / 'nnUNet_raw_data' / args.task

    try:
        taskdir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f'Destination "{taskdir}" already exists')
        return

    # Determine image source and destination directories
    image_srcdir = Path(args.images)
    if '*' in image_srcdir.name:
        image_glob_pattern = image_srcdir.name
        image_srcdir = image_srcdir.parent
    else:
        image_glob_pattern = '*.mha'

    image_dstdir = taskdir / 'imagesTr'
    image_dstdir.mkdir()

    # Prepare mask source and destianation directories
    mask_srcdir = Path(args.masks)
    mask_dstdir = taskdir / 'labelsTr'
    mask_dstdir.mkdir()

    # Copy / convert images and masks
    print('[#] Converting images and masks')
    training = []
    for image_srcfile in sorted(image_srcdir.glob(image_glob_pattern)):
        if image_srcfile.name.startswith('.'):
            # For some reason, hidden files are sometimes included and Mac users run
            # into problems then because they have plenty of ._ metadata files.
            continue

        if image_srcfile.name.endswith('.nii.gz'):
            caseid = image_srcfile.name[:-7]
            ext = 'nii.gz'
        else:
            caseid = image_srcfile.stem
            ext = image_srcfile.suffix[1:]

        if caseid.endswith('_0000'):
            caseid = caseid[:-5]

        try:
            mask_srcfile = mask_srcdir / f'{caseid}.{ext}'
            if not mask_srcfile.exists():
                mask_srcfile = next(mask_srcdir.glob(f'{caseid}_*.{ext}'))
        except StopIteration:
            print(f'Missing mask for case "{caseid}"')
            return

        image_dstfile = image_dstdir / f'{caseid}_0000.nii.gz'
        print(f'{image_srcfile.name} -> {image_dstfile.name}')
        atomic_file_copy(image_srcfile, image_dstfile)

        mask_dstfile = mask_dstdir / f'{caseid}.nii.gz'
        atomic_file_copy(mask_srcfile, mask_dstfile)

        training.append({
            'image': f'./imagesTr/{caseid}.nii.gz',
            'label': f'./labelsTr/{caseid}.nii.gz'
        })

    # Create metadata file
    print('[#] Writing metadata to dataset.json')

    name = args.task.split('_', 1)[1]
    labels = OrderedDict([(str(i), label) for i, label in enumerate(args.labels)])
    metadata = OrderedDict([
        ('name', name),
        ('description', f'{name}, reformatted for nnU-net'),
        ('tensorImageSize', '3D'),
        ('licence', args.license),
        ('release', args.release),
        ('modality', {'0': args.modality}),
        ('labels', labels),
        ('numTraining', len(training)),
        ('numTest', 0),
        ('training', training),
        ('test', [])
    ])
    write_json(taskdir / 'dataset.json', metadata, make_dirs=False)


def plan_train(argv):
    # Plan experiment, then train network
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('--results', type=str, required=False)
    parser.add_argument('--network', type=str, default='3d_fullres')
    parser.add_argument('--trainer', type=str, default='nnUNetTrainerV2')
    parser.add_argument('--trainer_kwargs', required=False, default="{}",
                        help="Use a dictionary in string format to specify keyword arguments. This will get"
                             " parsed into a dictionary, the values get correctly parsed to the data format"
                             " and passed to the trainer. Example (backslash included): \n"
                             r"--trainer_kwargs='{\"class_weights\":[0,2.00990337,1.42540704,2.13387239,0.85529504,0.592059,0.30040984,8.26874351],"
                             r"\"weight_dc\":0.3,\"weight_ce\":0.7}'")
    parser.add_argument('--kwargs', type=str, required=False, default=None,
                        help="Specify additional arguments for nnUNet_train. Example (backslash included): \n"
                             r"--kwargs='--disable_postprocessing_on_folds'")
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--custom_split', type=str, help='Path to a JSON file with a custom data split into five folds')
    parser.add_argument('--plan_only', action='store_true', help='Run the planning step, but not the training step')
    parser.add_argument('--validation_only', action='store_true',
                        help='Do no run network training, only the final validation step')
    parser.add_argument('--ensembling', action='store_true',
                        help='Export probability maps for ensembling during the final validation')
    parser.add_argument('--use_compressed_data', action='store_true',
                        help='Disable unpacking of compressed training data, use with caution')
    parser.add_argument('--plan_2d', action='store_true', help='Enable planning of 2D experiments')
    parser.add_argument('--dont_plan_3d', action='store_true', help='Disable planning of 3D experiments')
    parser.add_argument('--carbontracker', action='store_true', help='Enables tracking of energy consumption')
    parser.add_argument('--pretrained_weights', type=str, required=False, default=None)
    args = parser.parse_args(argv)

    # aid type hinting
    args.task = str(args.task)

    # Set environment variables
    datadir = Path(args.data)
    prepdir = Path(os.environ.get('prepdir', '/home/user/data'))

    splits_file = prepdir / args.task / 'splits_final.pkl'

    os.environ['nnUNet_raw_data_base'] = str(datadir)
    os.environ['nnUNet_preprocessed'] = str(prepdir)
    os.environ['RESULTS_FOLDER'] = args.results if args.results else str(datadir / 'results')

    # Start carbontracker
    with CustomizedCarbonTracker(prepdir / 'carbontracker', enabled=args.carbontracker):
        # Check if plans and preprocessed data are available
        taskid = get_task_id(args.task)
        taskdir = datadir / 'nnUNet_preprocessed' / args.task

        if path_exists(taskdir) or path_exists(prepdir / args.task):
            # Found plans and preprocessed data (maybe need to copy to local node)
            if args.custom_split:
                splits_file = taskdir / 'splits_final.json'
                if not splits_file.exists():
                    splits_file = prepdir / args.task / 'splits_final.json'
                if not splits_file.exists() or checksum(splits_file) != checksum(args.custom_split):
                    print(f"[#] Found plans and preprocessed data for {args.task}"
                          " - but you also provided a custom split which is different"
                          " from the present split, this is not permitted")
                    return

            if args.plan_only:
                print(f'[#] Found plans and preprocessed data for {args.task} - nothing to do')
            else:
                print(f'[#] Found plans and preprocessed data for {args.task}')
                if not (prepdir / args.task).exists():
                    print("[#] Copying plans and preprocessed data to compute node")
                    prepdir.mkdir(parents=True, exist_ok=True)
                    shutil_sol.copytree(taskdir, prepdir / args.task)
                    print(f'[#] Copied plans and preprocessed data to compute node')
        else:
            # Plans and data not available yet, run preprocessing
            print('[#] Creating plans and preprocessing data')
            cmd = [
                'nnUNet_plan_and_preprocess',
                '-t', taskid,
                '-tl', os.environ.get("nnUNet_tl", '8'), '-tf', os.environ.get("nnUNet_tf", '8'),
                '--verify_dataset_integrity'
            ]
            if not args.plan_2d and '2d' not in args.network:
                cmd.extend(['--planner2d', 'None'])  # disable 2D planning to speed up the preprocessing phase
            if args.dont_plan_3d and '3d' not in args.network:
                cmd.extend(['--planner3d', 'None'])
            if args.pretrained_weights is not None:
                cmd.extend(['-pretrained_weights', args.pretrained_weights])
            subprocess.check_call(cmd)

            # Use a custom data split?
            if args.custom_split:
                splits = []
                for split in read_json(args.custom_split):
                    splits.append(OrderedDict([
                        ('train', np.array(split['train'])),
                        ('val', np.array(split['val']))
                    ]))

                splits_file.parent.mkdir(parents=True, exist_ok=True)
                with splits_file.open('wb') as fp:
                    pickle.dump(splits, fp)
                shutil_sol.copyfile(args.custom_split, splits_file.with_suffix('.json'))

            if (prepdir / args.task).absolute() != taskdir.absolute():
                # Copy preprocessed data to storage server
                print('[#] Copying plans and preprocessed data from compute node to storage server')
                taskdir.parent.mkdir(parents=True, exist_ok=True)
                shutil_sol.copytree(prepdir / args.task, taskdir)

        if args.plan_only:
            return

        # Run training
        cmd = [
            'nnUNet_train',
            args.network,
            args.trainer,
            taskid,
            args.fold
        ]

        fold_name = 'all' if args.fold == 'all' else f'fold_{args.fold}'
        outdir = Path(
            os.environ['RESULTS_FOLDER']) / 'nnUNet' / args.network / args.task / f'{args.trainer}__{PLANS}' / fold_name

        if args.validation_only:
            print('[#] Running validation step only')
            cmd.append('--validation_only')
        elif path_exists(outdir) and any(outdir.glob("*.model")):
            print('[#] Resuming network training')
            cmd.append('-c')
        else:
            print('[#] Starting network training')

        if args.trainer_kwargs:
            cmd.append('--trainer_kwargs=%s' % args.trainer_kwargs)
        if args.use_compressed_data:
            cmd.append('--use_compressed_data')
        if args.ensembling:
            cmd.append('--npz')
        if args.kwargs is not None:
            cmd.extend(args.kwargs.split(" "))
        print(f'[#] Running {" ".join(cmd)}')

        subprocess.check_call(cmd)

        # Copy split file since that is for sure available now (nnUNet_train has created
        # it if the file did not exist already - unless training with "all", so still check)
        if splits_file.exists() and splits_file.parent.absolute() != taskdir.absolute():
            shutil_sol.copyfile(splits_file, taskdir)


def reveal_split(argv):
    # Print out the 5-fold cross validation split
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('data', type=str)
    args = parser.parse_args(argv)

    # Locate split file
    datadir = Path(args.data)
    split_file = datadir / 'nnUNet_preprocessed' / args.task / 'splits_final.pkl'

    print_split_per_fold(split_file)


def find_best_configuration(argv):
    # Find best configuration by analyzing cross-validation results
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('--networks', type=str, nargs='*', default=['3d_fullres'])
    parser.add_argument('--trainer', type=str, default='nnUNetTrainerV2')
    args = parser.parse_args(argv)

    # Set environment variables
    datadir = Path(args.data)
    prepdir = datadir / 'nnUNet_preprocessed'
    os.environ['nnUNet_preprocessed'] = str(prepdir)
    os.environ['RESULTS_FOLDER'] = str(datadir / 'results')

    # Prepare output directory to prevent crashes
    print('[#] Preparing output directory')
    cmdir = datadir / 'results' / 'nnUNet' / 'ensembles' / args.task
    cmdir.mkdir(parents=True, exist_ok=True)

    # Run scripts
    for network in args.networks:
        print(f'[#] Consolidating folds for {network} network')
        subprocess.check_call([
            'nnUNet_determine_postprocessing',
            '-m', network,
            '-t', get_task_id(args.task),
            '-tr', str(args.trainer)
        ])

    if len(args.networks) > 1:
        print('[#] Finding best configuration')
        refresh_file_list(prepdir / args.task / 'gt_segmentations')
        subprocess.check_call([
            'nnUNet_find_best_configuration',
            '-m', *args.networks,
            '-t', get_task_id(args.task),
            '-tr', str(args.trainer)
        ])


def _predict(args):
    # Set environment variables
    os.environ['RESULTS_FOLDER'] = args.results

    # Prepare output directory to prevent crashes
    outdir = Path(args.output).absolute()
    outdir.mkdir(parents=True, exist_ok=True)

    # Run prediction script
    cmd = [
        'nnUNet_predict',
        '-t', args.task,
        '-i', args.input,
        '-o', args.output,
        '-m', args.network,
        '-tr', args.trainer,
        '--num_threads_preprocessing', '2',
        '--num_threads_nifti_save', '1'
    ]

    if args.folds:
        cmd.append('-f')
        cmd.extend(args.folds.split(','))

    if args.checkpoint:
        cmd.append('-chk')
        cmd.append(args.checkpoint)

    if args.store_probability_maps:
        cmd.append('--save_npz')

    if args.disable_augmentation:
        cmd.append('--disable_tta')

    if args.disable_patch_overlap:
        cmd.extend(['--step_size', '1'])

    subprocess.check_call(cmd)


def predict(argv):
    # Use trained network to generate segmentation masks
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('--input', type=str, default='/input')
    parser.add_argument('--output', type=str, default='/output')
    parser.add_argument('--results', type=str, required=True)  # Path to training results folder with model weights etc
    parser.add_argument('--network', type=str, default='3d_fullres')
    parser.add_argument('--trainer', type=str, default='nnUNetTrainerV2')
    parser.add_argument('--folds', type=str, required=False)
    parser.add_argument('--checkpoint', type=str,
                        required=False)  # Checkpoint to load, defaults to "model_final_checkpoint"
    parser.add_argument('--store_probability_maps', action='store_true')
    parser.add_argument('--disable_augmentation', action='store_true')
    parser.add_argument('--disable_patch_overlap', action='store_true')
    args = parser.parse_args(argv)

    _predict(args)


def ensemble(argv):
    # Run inference and ensemble predictions
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('--input', type=str, default='/input')
    parser.add_argument('--output', type=str, default='/output')
    parser.add_argument('--results', type=str, required=True)  # Path to training results folder with model weights etc
    parser.add_argument('--networks', type=str, nargs='*', default=['3d_fullres'])
    parser.add_argument('--trainers', type=str, nargs='*', default=['nnUNetTrainerV2'])
    parser.add_argument('--folds', type=str, required=False)
    parser.add_argument('--checkpoint', type=str,
                        required=False)  # Checkpoint to load, defaults to "model_final_checkpoint"
    parser.add_argument('--disable_augmentation', action='store_true')
    parser.add_argument('--disable_patch_overlap', action='store_true')
    args = parser.parse_args(argv)

    # Run inference for all networks in the ensemble
    output_dirs = []
    ensemble_name_fragments = []
    for i, network in enumerate(args.networks):
        print(f'[#] Running inference for {network} network')

        args_predict = deepcopy(args)
        args_predict.store_probability_maps = True

        args_predict.network = network
        del args_predict.networks

        args_predict.trainer = args.trainers[i] if len(args.trainers) > i else args.trainers[-1]
        del args_predict.trainers

        output_dir = Path(args.output) / network
        output_dirs.append(output_dir)
        args_predict.output = str(output_dir)

        ensemble_name_fragments.append(f'{args_predict.network}__{args_predict.trainer}__{PLANS}')
        _predict(args_predict)

    # Combine results
    print('[#] Ensembling results')
    ensemble_name = 'ensemble_' + '--'.join(ensemble_name_fragments)
    output_dir = Path(args.output) / ensemble_name

    cmd = [
        'nnUNet_ensemble',
        '-f', *[str(f) for f in output_dirs],
        '-o', str(output_dir)
    ]

    pp_file = Path(args.results) / 'nnUNet' / 'ensembles' / args.task / ensemble_name / 'postprocessing.json'
    if path_exists(pp_file):
        cmd.extend(['-pp', str(pp_file)])

    subprocess.check_call(cmd)


def evaluate(argv):
    # Run evaluation on the test set
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth', type=str, required=True)
    parser.add_argument('--prediction', type=str, required=True)
    parser.add_argument('--labels', type=str, nargs='+', required=True)
    args = parser.parse_args(argv)

    # Check if the specified folders exist
    ground_truth_dir = Path(args.ground_truth)
    if not path_exists(ground_truth_dir):
        print('Folder with ground truth annotations does not exist')
        return

    prediction_dir = Path(args.prediction)
    if not path_exists(prediction_dir):
        print('Folder with ground truth annotations does not exist')
        return

    # Parse labels
    range_pattern = re.compile('[0-9]+-[0-9]+')
    if len(args.labels) == 1 and range_pattern.fullmatch(args.labels[0]):
        r = tuple(map(int, args.labels[0].split('-')))
        labels = list(map(str, range(r[0], r[1] + 1)))
    else:
        labels = args.labels

    # Run nnUNet script
    print('[#] Evaluating test set results')
    subprocess.check_call([
        'nnUNet_evaluate_folder',
        '-ref', str(ground_truth_dir),
        '-pred', str(prediction_dir),
        '-l', *labels
    ])

    # Read results file
    results_file = prediction_dir / 'summary.json'
    try:
        results = read_json(results_file)
    except IOError:
        print('Evaluation failed')
        return

    print('Average Dice scores across all cases:')
    for label, metrics in sorted(results['results']['mean'].items(), key=lambda item: int(item[0])):
        print(f' > {label}: {metrics["Dice"]}')


def checkout(argv):
    # Switch to a specific branch of the nnU-Net repository?
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', type=str, default='')
    args, unknown = parser.parse_known_args(argv)

    if args.checkout:
        subprocess.check_call([
            'git', '-C', '/home/user/nnunet',
            'fetch'
        ])
        subprocess.check_call([
            'git', '-C', '/home/user/nnunet',
            'checkout', args.checkout
        ])

    return unknown


if __name__ == '__main__':
    # Very first argument determines action
    actions = {
        'prepare': prepare,
        'plan_train': plan_train,
        'reveal_split': reveal_split,
        'find_best_configuration': find_best_configuration,
        'predict': predict,
        'ensemble': ensemble,
        'evaluate': evaluate
    }

    try:
        action = actions[sys.argv[1]]
        argv = checkout(sys.argv[2:])
    except (IndexError, KeyError):
        print('Usage: nnunet ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(argv)
