docker run --cpus=8 --memory=32g --shm-size=32g --rm -it \
    -v ~/joeran/repos/picai_baseline/src/picai_baseline/nnunet_semi_supervised/training_sagemaker/code:/code \
    -v /media/pelvis/projects/joeran/picai/debug-workdir:/workdir \
    -v /media/pelvis/data/prostate-MRI/picai/debug:/input \
    -v /media/pelvis/projects/joeran/picai/debug-output:/input/preprocessed \
    joeranbosma/picai_nnunet:latest \
    python /code/train.py
