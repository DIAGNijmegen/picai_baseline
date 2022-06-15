[← Return to overview](https://github.com/DIAGNijmegen/picai_baseline#baseline-ai-models-for-prostate-cancer-detection-in-mri)

#

## nnDetection
The nnDetection framework is geared towards medical object detection [[2]](https://github.com/DIAGNijmegen/picai_baseline#2). Setting up nnDetection and tweaking its implementation is not as straightforward as for the [nnUNet](nnunet_baseline.md) or [UNet](unet_baseline.md) baselines, but it can provide a strong csPCa detection model. Interested readers who would like to modify the implementation of nnDetection are referred to the [nnDetection documentation](https://github.com/MIC-DKFZ/nnDetection/). We only provide training and evaluation steps with the vanilla nnDetection framework.


### nnDetection - Docker Setup
To run nnDetection commands, you can use the Docker specified in [`nndetection/training_docker/`](src/picai_baseline/nndetection/training_docker/). This is a wrapper around nnDetection, and facilitates training in a Docker container on a distributed system.

To build the Docker container, navigate to [`nndetection/training_docker/`](src/picai_baseline/nndetection/training_docker/) and build the container:

```
cd src/picai_baseline/nndetection/training_docker/
docker build . --tag joeranbosma/picai_nndetection:latest
```

This will result (if ran successfully) in the Docker container named `joeranbosma/picai_nndetection:latest`. Alternatively, the pre-built Docker container can be loaded:

```
docker pull joeranbosma/picai_nndetection:latest
```


### nnDetection - Data Preparation
We use the [nnUNet Raw Data Archive][nnunet-archive] format as starting point, as obtained after the steps in [Data Preprocessing](https://github.com/DIAGNijmegen/picai_baseline#data-preprocessing). Because all lesions are non-touching, the nnUNet Raw Data Archive can be converted to the nnDetection Raw Data Archive format unabmiguously. After finishing the steps described in [Data Preprocessing](https://github.com/DIAGNijmegen/picai_baseline#data-preprocessing), the nnUNet Raw Data Archive can be converted into the nnDetection raw data archive using the following command:

```bash
python -m picai_prep nnunet2nndet \
    --input /workdir/nnUNet_raw_data/Task2201_picai_baseline \
    --output /workdir/nnDet_raw_data/Task2201_picai_baseline
```

Alternatively, you can use Docker to run the Python script:

```bash
docker run --cpus=2 --memory=16gb --rm \
    -v /path/to/workdir/:/workdir/ \
    joeranbosma/picai_nnunet:latest python3 -m picai_prep nnunet2nndet --input /workdir/nnUNet_raw_data/Task2201_picai_baseline --output /workdir/nnDet_raw_data/Task2201_picai_baseline
```

nnDetection also requires user-defined cross-validation splits to ensure there is no patient overlap between training and validation splits. The official cross-validation splits can be stored to the working directory using the steps in [nnU-Net - Cross-Validation Splits](nnunet_baseline.md#nnu-net---cross-validation-splits).


### nnDetection - Training
Running the first fold will start with preprocessing the raw images. After preprocessing is done, it will automatically start training.

```bash
docker run --cpus=8 --memory=32gb --shm-size=32gb --gpus='"device=6"' -it --rm \
    -v /path/to/workdir:/workdir \
    joeranbosma/picai_nndetection:latest nndet prep_train \
    Task2201_picai_baseline /workdir/ \
    --custom_split /workdir/nnUNet_raw_data/Task2201_picai_baseline/splits.json \
    --fold 0
```

After preprocessing is done, the other folds can be run sequentially or in parallel with the first fold (change to `--fold 1`, etc.)

Note: runs in our environment with 32 GB RAM, 8 CPUs, 1 GPU with 8 GB VRAM. Takes about 1 day per fold on a RTX 2080 Ti.


### nnDetection - Inference 🏗
_Under construction._


### nnDetection - Evaluation 🏗
_Under construction._
<!--Please follow the steps in [nnUNet - extract lesion candidates](nnunet_baseline.md#nnUNet---extract-lesion-candidates).-->


### nnDetection - Algorithm Submission 🏗
_Under construction._


## References
<a id="1" href="https://www.nature.com/articles/s41592-020-01008-z">[1]</a> 
Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen and Klaus H. Maier-Hein. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation". Nature Methods 18.2 (2021): 203-211.

<a id="2" href="https://link.springer.com/chapter/10.1007/978-3-030-87240-3_51">[2]</a> 
Michael Baumgartner, Paul F. Jaeger, Fabian Isensee, Klaus H. Maier-Hein. "nnDetection: A Self-configuring Method for Medical Object Detection". International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2021.

<a id="3" href="https://arxiv.org/abs/2112.05151">[3]</a> 
Joeran Bosma, Anindo Saha, Matin Hosseinzadeh, Ilse Slootweg, Maarten de Rooij, Henkjan Huisman. "Semi-supervised learning with report-guided lesion annotation for deep learning-based prostate cancer detection in bpMRI". arXiv:2112.05151.

<a id="4" href="#">[4]</a> 
Joeran Bosma, Natalia Alves and Henkjan Huisman. "Performant and Reproducible Deep Learning-Based Cancer Detection Models for Medical Imaging". _Under Review_.

[nnunet-archive]: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md
