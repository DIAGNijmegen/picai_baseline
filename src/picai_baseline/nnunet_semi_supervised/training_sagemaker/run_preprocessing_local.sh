docker run --cpus=8 --memory=16g --shm-size=16g --rm -it \
    -v ~/joeran/repos/picai_baseline/src/picai_baseline/nnunet_semi_supervised/training_sagemaker/code:/code \
    -v /media/pelvis/projects/joeran/picai/debug-workdir:/workdir \
    -v /media/pelvis/data/prostate-MRI/picai/debug/images:/input/images \
    -v /media/pelvis/data/prostate-MRI/picai/debug/picai_labels:/input/picai_labels \
    -v /media/pelvis/data/prostate-MRI/picai/debug/scripts:/scripts \
    -v /media/pelvis/projects/joeran/picai/debug-output:/output \
    joeranbosma/picai_nnunet:latest \
    python /code/preprocess.py
