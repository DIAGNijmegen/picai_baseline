from multiprocessing import Pool
from pathlib import Path

from picai_baseline.nnunet.softmax_export import \
    convert_cropped_npz_to_original_nifty
from picai_baseline.splits.picai_nnunet import valid_splits
from picai_eval import evaluate_folder
from report_guided_annotation import extract_lesion_candidates

task = "Task2201_picai_baseline"
trainer = "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1"
task_dir = Path("/media/pelvis/projects/joeran/picai/workdir/results/nnUNet/3d_fullres/") / task
checkpoints = ["model_best"]


for fold in range(5):
    print(f"Fold {fold}")

    for checkpoint in checkpoints:
        softmax_dir = task_dir / trainer / f"fold_{fold}/picai_pubtrain_predictions_{checkpoint}"
        metrics_path = softmax_dir.parent  / f"metrics-{checkpoint}-picai_eval-dynamic.json"

        if metrics_path.exists():
            print(f"Metrics found at {metrics_path}, skipping..")
            continue

        # pad raw npz predictions to their original extent
        with Pool() as pool:
            pool.map(
                func=convert_cropped_npz_to_original_nifty,
                iterable=list(softmax_dir.glob("*.npz")),
            )

        # evaluate
        metrics = evaluate_folder(
            y_det_dir=softmax_dir,
            y_true_dir=f"/media/pelvis/projects/joeran/picai/workdir/nnUNet_raw_data/{task}/labelsTr",
            subject_list=valid_splits[fold]['subject_list'],
            pred_extensions=['_softmax.nii.gz'],
            y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred, threshold="dynamic")[0],
        )

        # save and show metrics
        metrics.save(metrics_path)
        print(f"Results for checkpoint {checkpoint}:")
        print(metrics)
