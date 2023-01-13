docker run --cpus=8 --memory=32g --shm-size=32g --rm -it \
    -v ~/joeran/repos/picai_baseline/src/picai_baseline/nnunet_semi_supervised/training_sagemaker/code:/code:ro \
    -v /media/pelvis/projects/joeran/picai/debug-workdir:/workdir \
    -v /media/pelvis/data/prostate-MRI/picai/public_training:/input:ro \
    -v /media/pelvis/projects/joeran/picai/debug-checkpoints:/checkpoints \
    -v /media/pelvis/projects/joeran/picai/debug-output:/output \
    joeranbosma/picai_nnunet:latest \
    python /code/preprocess.py --splits picai_pub
