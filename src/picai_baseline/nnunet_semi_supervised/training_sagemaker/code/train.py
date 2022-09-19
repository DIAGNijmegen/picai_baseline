import argparse
import os
import shutil
from pathlib import Path
from subprocess import check_call


def main():

    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--workdir', type=str, default="/workdir")
    parser.add_argument('--imagesdir', type=str, default=os.environ['SM_CHANNEL_IMAGES'])
    parser.add_argument('--labelsdir', type=str, default=os.environ['SM_CHANNEL_LABELS'])
    parser.add_argument('--scriptsdir', type=str, default=os.environ['SM_CHANNEL_SCRIPTS'])
    parser.add_argument('--outputdir', type=str, default=os.environ['SM_MODEL_DIR'])

    args, _ = parser.parse_known_args()

    # paths
    workdir = Path(args.workdir)
    images_dir = Path(args.imagesdir)
    labels_dir = Path(args.labelsdir)
    output_dir = Path(args.outputdir)
    scripts_dir = Path(args.scriptsdir)

    # print input paths
    print(f"workdir: {workdir}")
    print(f"images_dir: {images_dir}")
    print(f"labels_dir: {labels_dir}")
    print(f"output_dir: {output_dir}")
    print(f"scripts_dir: {scripts_dir}")

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(scripts_dir, "contains", os.listdir(scripts_dir))
    print(images_dir, "contains", os.listdir(images_dir))
    print(labels_dir, "contains", os.listdir(labels_dir))

    # install modified nnU-Net
    cmd = [
        "pip",
        "install",
        "-e",
        str(scripts_dir / "nnunet"),
    ]
    check_call(cmd)

    # Convert MHA Archive to nnU-Net Raw Data Archive
    # Also, we combine the provided human-expert annotations with the AI-derived annotations.
    print("Preprocessing data...")
    cmd = [
        "python",
        (scripts_dir / "picai_baseline/src/picai_baseline/prepare_data_semi_supervised.py").as_posix(),
        "--workdir", workdir.as_posix(),
        "--imagesdir", images_dir.as_posix(),
        "--labelsdir", labels_dir.as_posix(),
    ]
    check_call(cmd)

    # Train models
    folds = range(1)  # range(5) for 5-fold cross-validation
    for fold in folds:
        print(f"Training fold {fold}...")
        cmd = [
            "python", (scripts_dir / "picai_baseline/src/picai_baseline/nnunet/training_docker/nnunet_wrapper.py").as_posix(),
            "plan_train", "Task2203_picai_baseline", workdir.as_posix(),
            "--trainer", "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
            "--fold", str(fold),
        ]
        check_call(cmd)

    # Export trained models
    for fold in folds:
        src = workdir / f"results/nnUNet/3d_fullres/Task2203_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_{fold}/model_best.model"
        dst = output_dir / f"picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_{fold}/model_best.model"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

        src = workdir / f"results/nnUNet/3d_fullres/Task2203_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_{fold}/model_best.model.pkl"
        dst = output_dir / f"picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_{fold}/model_best.model.pkl"
        shutil.copy(src, dst)

    shutil.copy(workdir / "results/nnUNet/3d_fullres/Task2203_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/plans.pkl",
                output_dir / "picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/plans.pkl")


if __name__ == '__main__':
    main()
