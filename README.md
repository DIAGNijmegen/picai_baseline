# Baseline AI Models for Prostate Cancer Detection in MRI

This repository contains utilities to set up and train deep learning-based detection models for clinically significant prostate cancer (csPCa) in MRI. In turn, these models serve as the official baseline AI solutions for the [PI-CAI challenge](https://pi-cai.grand-challenge.org/). As of now, the following three models will be provided and supported:

- [U-Net](unet_baseline.md)
- [nnU-Net](nnunet_baseline.md)
- [nnDetection](nndetection_baseline.md)

All three solutions share the same starting point, with respect to their expected [folder structure](#folder-structure) and [data preparation](#data-preparation) pipeline.

## Issues
Please feel free to raise any issues you encounter [here](https://github.com/DIAGNijmegen/picai_baseline/issues).


## Installation
`picai_baseline` can be pip-installed:

```bash
pip install picai_baseline
```

Alternatively, `picai_baseline` can be installed from source:

```bash
git clone https://github.com/DIAGNijmegen/picai_baseline
cd picai_baseline
pip install -e .
```

This ensures the scripts are present locally, which enables you to run the provided Python scripts. Additionally, this allows you to modify the baseline solutions, due to the `-e` option. Furthermore, this ensures the latest version is installed.


## General Setup
We define setup steps that are shared between the different baseline algorithms. To follow the baseline algorithm tutorials, this setup must be completed first.


### Folder Structure
We define three main folders that must be prepared apriori:
- `/input/` contains one of the [PI-CAI datasets](https://pi-cai.grand-challenge.org/DATA/). This can be the Public Training and Development Dataset, the Private Training Dataset, the Hidden Validation and Tuning Cohort, or the Hidden Testing Cohort.
  - `/input/images/` contains the imaging files. For the Public Training and Development Dataset, these can be retrieved [here](https://zenodo.org/record/6624726).
  - `/input/labels/` contains the annotations. For the Public Training and Development Dataset, these can be retrieved [here](https://github.com/DIAGNijmegen/picai_labels).
- `/workdir/` stores intermediate results, such as preprocessed images and annotations.
  - `/workdir/results/[model name]/` stores model checkpoints/weights during training (enables the ability to pause/resume training).    
- `/output/` stores training output, such as trained model weights and preprocessing plan.


### Data Preparation
Unless specified otherwise, this tutorial assumes that the [PI-CAI: Public Training and Development Dataset](https://pi-cai.grand-challenge.org/DATA/) will be downloaded and unpacked. Before downloading the dataset, read its [documentation](https://zenodo.org/record/6624726) and [dedicated forum post](https://grand-challenge.org/forums/forum/pi-cai-607/topic/public-training-and-development-dataset-updates-and-fixes-631/) (for all updates/fixes, if any). To download and unpack the dataset, run the following commands:

```bash
# download all folds
curl -C - "https://zenodo.org/record/6624726/files/picai_public_images_fold0.zip?download=1" --output picai_public_images_fold0.zip
curl -C - "https://zenodo.org/record/6624726/files/picai_public_images_fold1.zip?download=1" --output picai_public_images_fold1.zip
curl -C - "https://zenodo.org/record/6624726/files/picai_public_images_fold2.zip?download=1" --output picai_public_images_fold2.zip
curl -C - "https://zenodo.org/record/6624726/files/picai_public_images_fold3.zip?download=1" --output picai_public_images_fold3.zip
curl -C - "https://zenodo.org/record/6624726/files/picai_public_images_fold4.zip?download=1" --output picai_public_images_fold4.zip

# unzip all folds
unzip picai_public_images_fold0.zip -d /input/images/
unzip picai_public_images_fold1.zip -d /input/images/
unzip picai_public_images_fold2.zip -d /input/images/
unzip picai_public_images_fold3.zip -d /input/images/
unzip picai_public_images_fold4.zip -d /input/images/
```

In case `unzip` is not installed, you can use Docker to unzip the files:

```bash
docker run --cpus=2 --memory=8gb --rm -v /path/to/input:/input joeranbosma/picai_nnunet:latest unzip /input/picai_public_images_fold0.zip -d /input/images/
docker run --cpus=2 --memory=8gb --rm -v /path/to/input:/input joeranbosma/picai_nnunet:latest unzip /input/picai_public_images_fold1.zip -d /input/images/
docker run --cpus=2 --memory=8gb --rm -v /path/to/input:/input joeranbosma/picai_nnunet:latest unzip /input/picai_public_images_fold2.zip -d /input/images/
docker run --cpus=2 --memory=8gb --rm -v /path/to/input:/input joeranbosma/picai_nnunet:latest unzip /input/picai_public_images_fold3.zip -d /input/images/
docker run --cpus=2 --memory=8gb --rm -v /path/to/input:/input joeranbosma/picai_nnunet:latest unzip /input/picai_public_images_fold4.zip -d /input/images/
```

Please follow the [instructions here](nnunet_baseline.md#nnu-net---docker-setup) to set up the Docker container.

Also, collect the training annotations via the following command:

```bash
git clone https://github.com/DIAGNijmegen/picai_labels /input/labels/
```


### Cross-Validation Splits
We have prepared 5-fold cross-validation splits of all 1500 cases in the [PI-CAI: Public Training and Development Dataset](https://pi-cai.grand-challenge.org/DATA/). We have ensured there is no patient overlap between training/validation splits. You can load these splits as follows:

```python
from picai_baseline.splits.picai import train_splits, valid_splits

for fold, ds_config in train_splits.items():
    print(f"Training fold {fold} has cases: {ds_config['subject_list']}")

for fold, ds_config in valid_splits.items():
    print(f"Validation fold {fold} has cases: {ds_config['subject_list']}")
```

Additionally, we prepared 5-fold cross-validation splits of all cases with an [expert-derived csPCa annotation](https://github.com/DIAGNijmegen/picai_labels/tree/main/csPCa_lesion_delineations/human_expert). These splits are subsets of the splits above. You can load these splits as follows:

```python
from picai_baseline.splits.picai_nnunet import train_splits, valid_splits
```

When using `picai_eval` from the command line, we recommend saving the splits to disk. Then, you can pass these to `picai_eval` to ensure all cases were found. You can export the labelled cross-validation splits using:

```bash
python -m picai_baseline.splits.picai_nnunet --output "/workdir/splits/picai_nnunet"
```


### Data Preprocessing
We follow the [`nnU-Net Raw Data Archive`][nnunet_raw_data_format] format to prepare our dataset for usage. For this, you can use the [`picai_prep`][picai_prep] module. Note, the [`picai_prep`][picai_prep] module should be automatically installed when installing the `picai_baseline` module, and is installed within the [`picai_nnunet`][picai_nnunet_docker] and [`picai_nndetection`][picai_nndetection_docker] Docker containers as well. 

To convert the dataset in `/input/` into the [`nnU-Net Raw Data Archive`][nnunet_raw_data_format] format, and store it in `/workdir/nnUNet_raw_data`, please follow the instructions [provided here][picai_prep_mha2nnunet], or set your target paths in [`prepare_data.py`](src/picai_baseline/prepare_data.py) and execute it:

```bash
python src/picai_baseline/prepare_data.py
```

To adapt/modify the preprocessing pipeline or its default specifications, please make changes to the [`prepare_data.py`](src/picai_baseline/prepare_data.py) script accordingly.

Alternatively, you can use Docker to run the Python script:

```bash
docker run --cpus=2 --memory=16gb --rm \
    -v /path/to/input/:/input/ \
    -v /path/to/workdir/:/workdir/ \
    -v /path/to/picai_baseline:/scripts/picai_baseline/ \
    joeranbosma/picai_nnunet:latest python3 /scripts/picai_baseline/src/picai_baseline/prepare_data.py
```


## Baseline Algorithms
We provide end-to-end training pipelines for csPCa detection/diagnosis in 3D. Each baseline includes a template to encapsulate the trained AI model in a Docker container, and uploading the same to the [grand-challenge.org](https://grand-challenge.org/) platform as an ["algorithm"](https://grand-challenge.org/documentation/algorithms/). 


### U-Net
We include a baseline [U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) to provide a playground environment for participants and kickstart their development cycle. The U-Net baseline generates quick results with minimal complexity, but does so at the expense of sub-optimal performance and low flexibility in adapting to any other task.

[→ Read the full documentation here](unet_baseline.md).


### nnU-Net
The nnU-Net framework [[1]](#1) provides a performant framework for medical image segmentation, which is straightforward to adapt for csPCa detection. 

[→ Read the full documentation here](nnunet_baseline.md).


### nnDetection
The nnDetection framework is geared towards medical object detection [[2]](#2). Setting up nnDetection and tweaking its implementation is not as straightforward as for the [nnUNet](#nnu-net) or [UNet](#u-net) baselines, but it can provide a strong csPCa detection model.

[→ Read the full documentation here](nndetection_baseline.md).


## References
<a id="1" href="https://www.nature.com/articles/s41592-020-01008-z">[1]</a> 
Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen and Klaus H. Maier-Hein. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation". Nature Methods 18.2 (2021): 203-211.

<a id="2" href="https://link.springer.com/chapter/10.1007/978-3-030-87240-3_51">[2]</a> 
Michael Baumgartner, Paul F. Jaeger, Fabian Isensee, Klaus H. Maier-Hein. "nnDetection: A Self-configuring Method for Medical Object Detection". International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2021.

<a id="3" href="https://arxiv.org/abs/2112.05151">[3]</a> 
Joeran Bosma, Anindo Saha, Matin Hosseinzadeh, Ilse Slootweg, Maarten de Rooij, Henkjan Huisman. "Semi-supervised learning with report-guided lesion annotation for deep learning-based prostate cancer detection in bpMRI". arXiv:2112.05151.

<a id="4" href="#">[4]</a> 
Joeran Bosma, Natalia Alves and Henkjan Huisman. "Performant and Reproducible Deep Learning-Based Cancer Detection Models for Medical Imaging". _Under Review_.


##
If you are using this codebase or some part of it, please cite the following article:

● [A. Saha, J. J. Twilt, J. S. Bosma, B. van Ginneken, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, "Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge (Study Protocol)", DOI: 10.5281/zenodo.6667655](https://zenodo.org/record/6667655)

**BibTeX:**
```
@ARTICLE{PICAI_BIAS,
    author = {Anindo Saha, Jasper J. Twilt, Joeran S. Bosma, Bram van Ginneken, Derya Yakar, Mattijs Elschot, Jeroen Veltman, Jurgen Fütterer, Maarten de Rooij, Henkjan Huisman},
    title  = {{Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge (Study Protocol)}}, 
    year   = {2022},
    doi    = {10.5281/zenodo.6667655}
}
```

[picai_nnunet_docker]: https://hub.docker.com/r/joeranbosma/picai_nnunet
[picai_nndetection_docker]: https://hub.docker.com/r/joeranbosma/picai_nndetection
[picai_prep]: https://github.com/DIAGNijmegen/picai_prep
[nnunet_raw_data_format]: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md
[picai_prep_mha2nnunet]: https://github.com/DIAGNijmegen/picai_prep#mha-archive--nnu-net-raw-data-archive
[nnunet-archive]: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md
[nndetection-archive]: https://github.com/MIC-DKFZ/nnDetection/#adding-new-data-sets
