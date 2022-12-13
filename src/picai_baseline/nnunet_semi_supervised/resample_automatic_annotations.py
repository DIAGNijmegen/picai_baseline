import argparse
from pathlib import Path
from typing import Union

import SimpleITK as sitk
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import resample_to_reference_scan
from tqdm import tqdm

from picai_baseline.splits.picai_pubpriv import valid_splits


def resample_annotations(
    task: str,
    trainer: str,
    mountpoint: Union[Path, str],
    workdir: Union[Path, str],
    in_dir_data: Union[Path, str],
    prediction_folder_name: str = "picai_pubpriv_predictions_ensemble_model_best_automatic_annotations"
):
    # paths
    mountpoint = Path(mountpoint)
    workdir = mountpoint / workdir
    in_dir_data = mountpoint / in_dir_data

    in_dir_ava = workdir / f"results/nnUNet/3d_fullres/{task}/{trainer}__nnUNetPlansv2.1/{prediction_folder_name}"
    in_dir_annot = in_dir_data / "picai_labels/csPCa_lesion_delineations/human_expert/resampled"
    in_dir_scans = in_dir_data / "images"
    out_dir_ava = workdir / f"results/nnUNet/3d_fullres/{task}/{trainer}__nnUNetPlansv2.1/{prediction_folder_name}_resampled"
    
    out_dir_ava.mkdir(parents=True, exist_ok=True)

    for fold, split in valid_splits.items():
        for subject_id in tqdm(split['subject_list'], desc=f"Fold {fold}"):
            # paths
            lbl_manual_path = in_dir_annot / f"{subject_id}.nii.gz"
            lbl_ava_path = in_dir_ava / f"{subject_id}_softmax.nii.gz"
            t2_path = in_dir_scans / subject_id.split("_")[0] / f"{subject_id}_t2w.mha"
            out_path_ava = out_dir_ava / f"{subject_id}.nii.gz"

            if out_path_ava.exists():
                print(f"Resampled annotation for {subject_id} exists, skipping..")
                continue

            # read
            lbl_ava = sitk.ReadImage(str(lbl_ava_path))
            if lbl_manual_path.exists():
                reference = sitk.ReadImage(str(lbl_manual_path))
            else:
                reference = sitk.ReadImage(str(t2_path))

            # resample
            lbl = resample_to_reference_scan(lbl_ava, reference_scan_original=reference)

            # save
            atomic_image_write(lbl, out_path_ava)


if __name__ == '__main__':
    # setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Task2201_picai_baseline", help="The name of the task.")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainerV2_Loss_FL_and_CE_checkpoints", help="The name of the trainer.")
    parser.add_argument("--mountpoint", type=str, default="/media", help="The path to the mountpoint.")
    parser.add_argument("--workdir", type=str, default="pelvis/projects/joeran/picai/pubpriv-workdir", help="The path to the workdir (relative to the mountpoint).")
    parser.add_argument("--in-dir-data", type=str, default="pelvis/data/prostate-MRI/picai/pubpriv_training/v1", help="The path to the input data directory (relative to the mountpoint).")
    parser.add_argument("--prediction_folder_name", type=str, default="picai_pubpriv_predictions_ensemble_model_best_automatic_annotations")
    args = parser.parse_args()

    resample_annotations(
        task=args.task,
        trainer=args.trainer,
        mountpoint=args.mountpoint,
        workdir=args.workdir,
        in_dir_data=args.in_dir_data,
        prediction_folder_name=args.prediction_folder_name,
    )
