import argparse
import functools
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nndet.io import load_pickle, save_json
from nndet.utils.info import maybe_verbose_iterable
from picai_prep.data_utils import atomic_image_write

# add a flush to each print statement to ensure the stuff gets logged on SOL
print = functools.partial(print, flush=True)


def boxes2det(in_dir_pred, out_dir_det, target_label=None, threshold=0.0, min_num_voxels=10, verbose=1):
    """
    Convert bounding boxes to non-overlapping detection maps

    Adapted from https://github.com/MIC-DKFZ/nnDetection/blob/main/scripts/utils.py
    """
    # setup
    in_dir_pred = Path(in_dir_pred)
    out_dir_det = Path(out_dir_det)
    out_dir_det.mkdir(exist_ok=True)

    if verbose:
        print(f"""
        Generating detection maps from nnDetection boxes
        input: {in_dir_pred}
        output: {out_dir_det}
        minimum confidence: {threshold}
        minimum number of voxels: {min_num_voxels}
        """)

    case_ids = [p.stem.rsplit('_', 1)[0] for p in in_dir_pred.glob("*_boxes.pkl")]
    if not case_ids:
        raise ValueError("No boxes found in input directory")
    else:
        print(f"Found {len(case_ids)} cases") if verbose else None

    y_det = {}
    prediction_meta = {}
    for cid in maybe_verbose_iterable(case_ids):
        case_prediction_meta = {}
        res = load_pickle(in_dir_pred / f"{cid}_boxes.pkl")

        instance_mask = np.zeros(res["original_size_of_raw_data"], dtype=float)

        boxes = res["pred_boxes"]
        scores = res["pred_scores"]
        labels = res["pred_labels"]

        _mask = scores >= threshold
        boxes = boxes[_mask]
        labels = labels[_mask]
        scores = scores[_mask]

        idx = np.argsort(scores)[::-1]
        scores = scores[idx]
        boxes = boxes[idx]
        labels = labels[idx]

        for instance_id, (pbox, pscore, plabel) in enumerate(zip(boxes, scores, labels), start=1):
            # check if predicted box is for the target/cancer class
            if target_label is None:
                assert plabel == 0
            elif target_label != plabel:
                continue

            case_prediction_meta[instance_id] = {
                "score": float(pscore),
                "label": int(plabel),
                "box": list(map(int, pbox))
            }

            # construct 3D box of voxels for bounding box
            mask_slicing = [
                slice(int(pbox[0]) + 1, int(pbox[2])),
                slice(int(pbox[1]) + 1, int(pbox[3])),
            ]
            neighbourhood_slicing = [
                slice(int(pbox[0]), int(pbox[2]) + 1),
                slice(int(pbox[1]), int(pbox[3]) + 1),
            ]
            if instance_mask.ndim == 3:
                mask_slicing.append(slice(int(pbox[4]) + 1, int(pbox[5])))
                neighbourhood_slicing.append(slice(int(pbox[4]), int(pbox[5]) + 1))

            mask_slicing = tuple(mask_slicing)
            neighbourhood_slicing = tuple(neighbourhood_slicing)

            # check size of lesion candidate
            num_voxels = instance_mask[mask_slicing].size
            if num_voxels <= min_num_voxels:
                continue

            # check if adding lesion candidate would collide with prior (=higher confidence) lesion candidate
            if np.max(instance_mask[neighbourhood_slicing]) == 0:
                print(f"Setting box {mask_slicing} to p={pscore}") if verbose >= 2 else None
                instance_mask[mask_slicing] = pscore
            else:
                print(f"Have overlap! Previous p={np.max(instance_mask[mask_slicing]):.4f}") if verbose >= 2 else None

        # convert to SimpleITK
        instance_mask_itk = sitk.GetImageFromArray(instance_mask)
        instance_mask_itk.SetOrigin(res["itk_origin"])
        instance_mask_itk.SetDirection(res["itk_direction"])
        instance_mask_itk.SetSpacing(res["itk_spacing"])

        # save results
        atomic_image_write(instance_mask_itk, str(out_dir_det / f"{cid}_detection_map.nii.gz"))
        save_json(case_prediction_meta, out_dir_det / f"{cid}_boxes.json")
        y_det[cid] = instance_mask_itk
        prediction_meta[cid] = case_prediction_meta

    if verbose:
        print("Finished.")

    return prediction_meta, y_det


def main():
    # acquire and parse input and output paths
    parser = argparse.ArgumentParser(description='Generate detection maps from nnDetection boxes')
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to folder with model predicitons (nnDetection boxes)")
    parser.add_argument("-o", "--output", type=str, required=False,
                        help="Path to store detection maps. Default: input directory + _detection_maps")
    parser.add_argument("-l", "--target_label", type=int, required=False,
                        help="Class to create detection maps for. Default: 0, asserting no other classes exist.")
    args = parser.parse_args()

    # convert paths to Path
    args.input = Path(args.input)
    if args.output is None:
        args.output = args.input.with_name(args.input.name + "_detection_maps")
    else:
        args.output = Path(args.output)

    # perform conversion
    boxes2det(
        in_dir_pred=args.input,
        out_dir_det=args.output,
        target_label=args.target_label
    )


if __name__ == "__main__":
    main()
