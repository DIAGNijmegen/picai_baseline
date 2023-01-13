docker run --cpus=8 --memory=61g --shm-size=61g --gpus='"device=0"' --rm -it \
    -v /mnt/netcache/pelvis/projects/joeran/repos/picai_baseline/src/picai_baseline/nnunet_semi_supervised/training_sagemaker/code:/code \
    -v /mnt/netcache/pelvis/projects/joeran/picai/debug-workdir-training:/workdir \
    -v /mnt/netcache/pelvis/data/prostate-MRI/picai/debug:/input \
    -v /mnt/netcache/pelvis/projects/joeran/picai/debug-output:/input/preprocessed \
    joeranbosma/picai_nnunet:latest \
    python /code/train.py
