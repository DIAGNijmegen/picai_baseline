from picai_baseline.splits import export_splits
from picai_baseline.splits.picai import train_splits, valid_splits

if __name__ == "__main__":
    export_splits(
        train_splits=train_splits,
        valid_splits=valid_splits,
    )
