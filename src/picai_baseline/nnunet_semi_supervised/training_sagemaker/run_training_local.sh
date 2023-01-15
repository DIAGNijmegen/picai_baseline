docker run --cpus=8 --memory=61g --shm-size=61g --gpus='"device=0"' --rm -it \
    -v /mnt/netcache/pelvis/projects/joeran/repos/picai_baseline/src/picai_baseline/nnunet_semi_supervised/training_sagemaker/code:/code:ro \
    -v /mnt/netcache/pelvis/projects/joeran/picai/debug-workdir-training:/workdir \
    -v /mnt/netcache/pelvis/data/prostate-MRI/picai/debug/images:/input/images:ro \
    -v /mnt/netcache/pelvis/data/prostate-MRI/picai/debug/picai_labels:/input/picai_labels:ro \
    -v /mnt/netcache/pelvis/projects/joeran/picai/debug-preprocessed:/input/preprocessed:ro \
    -v /mnt/netcache/pelvis/projects/joeran/picai/debug-output:/output \
    -v /mnt/netcache/pelvis/projects/joeran/picai/debug-checkpoints:/checkpoints \
    joeranbosma/picai_nnunet:latest \
    python /code/train.py
