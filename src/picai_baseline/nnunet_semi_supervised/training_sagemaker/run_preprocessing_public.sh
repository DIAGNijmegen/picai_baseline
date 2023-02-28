docker run --cpus=8 --memory=32g --shm-size=32g --rm -it \
    -v /mnt/netcache/pelvis/projects/joeran/repos/picai_baseline/src/picai_baseline/nnunet_semi_supervised/training_sagemaker/code:/code:ro \
    -v /mnt/netcache/pelvis/projects/joeran/picai/debug-workdir:/workdir \
    -v /mnt/netcache/pelvis/data/prostate-MRI/picai/public_training/images:/input/images:ro \
    -v /mnt/netcache/pelvis/data/prostate-MRI/picai/public_training/picai_labels:/input/picai_labels:ro \
    -v /mnt/netcache/pelvis/projects/joeran/picai/debug-preprocessed:/output \
    joeranbosma/picai_nnunet:latest \
    python /code/preprocess.py --splits picai_pub
