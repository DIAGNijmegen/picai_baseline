from pathlib import Path

from picai_baseline.splits.picai_nnunet import valid_splits
from picai_eval import evaluate_folder
from picai_prep.preprocessing import crop_or_pad
from report_guided_annotation import extract_lesion_candidates


def crop(img):
    return crop_or_pad(img, size=(20, 320, 320))


task = "Task2201_picai_baseline"
trainer = "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1"
fold = 0
results_dir = Path("/workdir/results/nnUNet/3d_fullres/")

for epoch in range(50, 600+1, 50):
    checkpoint = f"model_ep_{epoch:03d}"
    softmax_dir = results_dir / task / trainer / f"fold_{fold}/picai_pubtrain_predictions_{checkpoint}"
    metrics_path = softmax_dir / "metrics.json"

    if metrics_path.exists():
        print(f"Metrics found at {metrics_path}, skipping..")

    # evaluate
    metrics = evaluate_folder(
        y_det_dir=softmax_dir,
        y_true_dir=f"/workdir/nnUNet_raw_data/{task}/labelsTr",
        subject_list=valid_splits[fold]['subject_list'],
        y_det_postprocess_func=lambda pred: crop(extract_lesion_candidates(pred)[0]),
        y_true_postprocess_func=crop,
    )

    # save and show metrics
    metrics.save(metrics_path)
    print(f"Results for epoch {epoch}:")
    print(metrics)
