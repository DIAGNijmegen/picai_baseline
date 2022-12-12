import json
from pathlib import Path

import pytest

from picai_baseline.splits import (subject_list_annotated,
                                   subject_list_annotated_pubpriv)
from picai_baseline.splits.picai import nnunet_splits as picai_nnunet_splits
from picai_baseline.splits.picai import train_splits as picai_train_splits
from picai_baseline.splits.picai import valid_splits as picai_valid_splits
from picai_baseline.splits.picai_debug import \
    nnunet_splits as picai_debug_nnunet_splits
from picai_baseline.splits.picai_debug import \
    train_splits as picai_debug_train_splits
from picai_baseline.splits.picai_debug import \
    valid_splits as picai_debug_valid_splits
from picai_baseline.splits.picai_nnunet import \
    nnunet_splits as picai_nnunet_nnunet_splits
from picai_baseline.splits.picai_nnunet import \
    train_splits as picai_nnunet_train_splits
from picai_baseline.splits.picai_nnunet import \
    valid_splits as picai_nnunet_valid_splits
from picai_baseline.splits.picai_pubpriv import \
    nnunet_splits as picai_pubpriv_nnunet_splits
from picai_baseline.splits.picai_pubpriv import \
    train_splits as picai_pubpriv_train_splits
from picai_baseline.splits.picai_pubpriv import \
    valid_splits as picai_pubpriv_valid_splits
from picai_baseline.splits.picai_pubpriv_nnunet import \
    nnunet_splits as picai_pubpriv_nnunet_nnunet_splits
from picai_baseline.splits.picai_pubpriv_nnunet import \
    train_splits as picai_pubpriv_nnunet_train_splits
from picai_baseline.splits.picai_pubpriv_nnunet import \
    valid_splits as picai_pubpriv_nnunet_valid_splits


@pytest.mark.parametrize("train_splits, valid_splits, nnunet_splits", [
    (picai_train_splits, picai_valid_splits, picai_nnunet_splits),
    (picai_debug_train_splits, picai_debug_valid_splits, picai_debug_nnunet_splits),
    (picai_nnunet_train_splits, picai_nnunet_valid_splits, picai_nnunet_nnunet_splits),
    (picai_pubpriv_train_splits, picai_pubpriv_valid_splits, picai_pubpriv_nnunet_splits),
    (picai_pubpriv_nnunet_train_splits, picai_pubpriv_nnunet_valid_splits, picai_pubpriv_nnunet_nnunet_splits),
])
def test_splits(train_splits, valid_splits, nnunet_splits):
    """Test that the splits are consistent and do not have overlap between validation folds."""
    folds = list(train_splits)
    for i in folds:
        for j in folds:
            if i == j:
                # assert no overlap between validation and training sets
                overlap = set(train_splits[i]["subject_list"]) & set(valid_splits[j]["subject_list"])
                assert len(overlap) == 0, f"Found overlap between training and validation set in fold {i}! Overlap = {overlap}"
            else:
                # assert no overlap between validation splits
                overlap = set(valid_splits[i]["subject_list"]) & set(valid_splits[j]["subject_list"])
                assert len(overlap) == 0, f"Found overlap between validation folds {i} and {j}! Overlap = {overlap}"

    # assert that the nnunet splits are consistent with the train/valid splits
    for i in folds:
        assert set(train_splits[i]["subject_list"]) == set(nnunet_splits[i]["train"]), f"Found mismatch between train_splits and nnunet_splits in fold {i}!"
        assert set(valid_splits[i]["subject_list"]) == set(nnunet_splits[i]["val"]), f"Found mismatch between valid_splits and nnunet_splits in fold {i}!"


@pytest.mark.parametrize("nnunet_splits, nnunet_splits_path", [
    (picai_nnunet_splits, "src/picai_baseline/splits/picai/splits.json"),
    (picai_debug_nnunet_splits, "src/picai_baseline/splits/picai_debug/splits.json"),
    (picai_nnunet_nnunet_splits, "src/picai_baseline/splits/picai_nnunet/splits.json"),
    (picai_pubpriv_nnunet_splits, "src/picai_baseline/splits/picai_pubpriv/splits.json"),
    (picai_pubpriv_nnunet_nnunet_splits, "src/picai_baseline/splits/picai_pubpriv_nnunet/splits.json"),
])
def test_nnunet_precomputed_split(nnunet_splits, nnunet_splits_path):
    """Test that the precomputed nnunet splits are consistent with the computed splits."""
    path = Path(__file__).parent.parent / nnunet_splits_path
    with open(path, "r") as f:
        precomputed_nnunet_splits = json.load(f)
    assert nnunet_splits == precomputed_nnunet_splits


@pytest.mark.parametrize("subject_list, expected_number", [
    (subject_list_annotated, 1295),
    (subject_list_annotated_pubpriv, 8215),
])
def test_number_of_annotated_cases(subject_list, expected_number):
    """Test that the number of annotated cases is as expected."""
    assert len(subject_list) == expected_number, f"Expected {expected_number} annotated cases, but found {len(subject_list)}!"


if __name__ == "__main__":
    pytest.main([__file__])
