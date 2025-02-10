# Baseline AI Models for Prostate Cancer Detection in MRI

This repository contains utilities to set up and train deep learning-based detection models for clinically significant prostate cancer (csPCa) in MRI. In turn, these models serve as the official baseline AI solutions for the [PI-CAI challenge](https://pi-cai.grand-challenge.org/). As of now, the following three models are provided and supported:

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

Installing from source ensures the scripts are present locally, which enables you to run the provided Python scripts. Additionally, this allows you to modify the baseline solutions, due to the `-e` option.


## General Setup
We define setup steps that are shared between the different baseline algorithms. To follow the model-specific baseline algorithm tutorials, these steps must be completed first.


### Folder Structure
We define three main folders that must be prepared:
- `/input/` contains the [PI-CAI dataset](https://pi-cai.grand-challenge.org/DATA/). In this tutorial we assume this is the PI-CAI Public Training and Development Dataset.
  - `/input/images/` contains the imaging files. For the Public Training and Development Dataset, these can be retrieved [here](https://zenodo.org/record/6624726).
  - `/input/picai_labels/` contains the annotations. For the Public Training and Development Dataset, these can be retrieved [here](https://github.com/DIAGNijmegen/picai_labels).
- `/workdir/` stores intermediate results, such as preprocessed images and annotations.
  - `/workdir/results/[model name]/` stores model checkpoints/weights during training (enables the ability to pause/resume training).    
- `/output/` stores training output, such as trained model weights and preprocessing plan.


### Data Preparation
Unless specified otherwise, this tutorial assumes that the [PI-CAI: Public Training and Development Dataset](https://pi-cai.grand-challenge.org/DATA/) will be downloaded and unpacked. Before downloading the dataset, read its [documentation](https://zenodo.org/record/6624726) and [dedicated forum post](https://grand-challenge.org/forums/forum/pi-cai-607/topic/public-training-and-development-dataset-updates-and-fixes-631/) (for all updates/fixes, if any). To download and unpack the dataset, run the following commands:

```bash
# download all folds
curl -C - "https://zenodo.org/api/records/6624726/files/picai_public_images_fold0.zip/content" --output picai_public_images_fold0.zip
curl -C - "https://zenodo.org/api/records/6624726/files/picai_public_images_fold1.zip/content" --output picai_public_images_fold1.zip
curl -C - "https://zenodo.org/api/records/6624726/files/picai_public_images_fold2.zip/content" --output picai_public_images_fold2.zip
curl -C - "https://zenodo.org/api/records/6624726/files/picai_public_images_fold3.zip/content" --output picai_public_images_fold3.zip
curl -C - "https://zenodo.org/api/records/6624726/files/picai_public_images_fold4.zip/content" --output picai_public_images_fold4.zip

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

Also, collect the training annotations. This can be done via the following command:

```bash
cd /input
git clone https://github.com/DIAGNijmegen/picai_labels
```

After cloning the repository with annotations, you should have a folder structure like this:

```bash
/input/picai_labels
├── anatomical_delineations
│   ├── ...
├── clinical_information
│   └── marksheet.csv
└── csPCa_lesion_delineations
    ├── ...
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
We follow the [`nnU-Net Raw Data Archive`][nnunet_raw_data_format] format to prepare our dataset for usage. For this, you can use the [`picai_prep`][picai_prep] module. The [`picai_prep`][picai_prep] module allows to resample all cases to the same resolution (you can resample each case indivudually to the same resolution between the different sequences, or choose to resample the full dataset to the same resolution). For details on the available options to convert the dataset in `/input/` into the [`nnU-Net Raw Data Archive`][nnunet_raw_data_format] format, and store it in `/workdir/nnUNet_raw_data`, please see the instructions [provided here][picai_prep_mha2nnunet]. Below we give the conversion as performed for the [baseline semi-supervised nnU-Net](https://grand-challenge.org/algorithms/pi-cai-pubpriv-nnu-net-semi-supervised/). For the U-Net baseline, please see the [U-Net tutorial](src/picai_baseline/unet_baseline.md#u-net---data-preparation) for extra instructions.

Note, the [`picai_prep`][picai_prep] module should be automatically installed when installing the `picai_baseline` module, and is installed within the [`picai_nnunet`][picai_nnunet_docker] and [`picai_nndetection`][picai_nndetection_docker] Docker containers as well. 

```bash
python src/picai_baseline/prepare_data_semi_supervised.py
```

For the baseline semi-supervised U-Net algorithm, specify the dataset-wise resolution: `--spacing 3.0 0.5 0.5`.
To adapt/modify the preprocessing pipeline or its default specifications, either check out the various command like options (use flag `-h` to show these) or make changes to the [`prepare_data_semi_supervised.py`](src/picai_baseline/prepare_data_semi_supervised.py) script.

Alternatively, you can use Docker to run the Python script:

```bash
docker run --cpus=2 --memory=16gb --rm \
    -v /path/to/input/:/input/ \
    -v /path/to/workdir/:/workdir/ \
    -v /path/to/picai_baseline:/scripts/picai_baseline/ \
    joeranbosma/picai_nnunet:latest python3 /scripts/picai_baseline/src/picai_baseline/prepare_data_semi_supervised.py
```

If you don't want to include the AI-generated annotations, you can also use the supervised data preparation script: [`prepare_data.py`](src/picai_baseline/prepare_data.py).


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

If you are using this codebase or some part of it, please cite the following article:

__[Saha A, Bosma JS, Twilt JJ, et al. Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study. Lancet Oncol 2024; 25: 879–887](https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(24)00220-1/fulltext)__  

If you are using the AI-generated annotations (i.e., semi-supervised learning), please cite the following article:

__[J. S. Bosma, A. Saha, M. Hosseinzadeh, I. Slootweg, M. de Rooij, and H. Huisman, "Semisupervised Learning with Report-guided Pseudo Labels for Deep Learning–based Prostate Cancer Detection Using Biparametric MRI", Radiology: Artificial Intelligence, 230031, 2023. doi:10.1148/ryai.230031](https://doi.org/10.1148/ryai.230031)__

**BibTeX:**
```
@ARTICLE{SahaBosmaTwilt2024,
  title = {Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study},
  journal = {The Lancet Oncology},
  year = {2024},
  issn = {1470-2045},
  volume={25},
  number={7},
  pages={879--887},
  doi = {https://doi.org/10.1016/S1470-2045(24)00220-1},
  author = {Anindo Saha and Joeran S Bosma and Jasper J Twilt and Bram {van Ginneken} and Anders Bjartell and Anwar R Padhani and David Bonekamp and Geert Villeirs and Georg Salomon and Gianluca Giannarini and Jayashree Kalpathy-Cramer and Jelle Barentsz and Klaus H Maier-Hein and Mirabela Rusu and Olivier Rouvière and Roderick {van den Bergh} and Valeria Panebianco and Veeru Kasivisvanathan and Nancy A Obuchowski and Derya Yakar and Mattijs Elschot and Jeroen Veltman and Jurgen J Fütterer and Constant R. Noordman and Ivan Slootweg and Christian Roest and Stefan J. Fransen and Mohammed R.S. Sunoqrot and Tone F. Bathen and Dennis Rouw and Jos Immerzeel and Jeroen Geerdink and Chris {van Run} and Miriam Groeneveld and James Meakin and Ahmet Karagöz and Alexandre Bône and Alexandre Routier and Arnaud Marcoux and Clément Abi-Nader and Cynthia Xinran Li and Dagan Feng and Deniz Alis and Ercan Karaarslan and Euijoon Ahn and François Nicolas and Geoffrey A. Sonn and Indrani Bhattacharya and Jinman Kim and Jun Shi and Hassan Jahanandish and Hong An and Hongyu Kan and Ilkay Oksuz and Liang Qiao and Marc-Michel Rohé and Mert Yergin and Mohamed Khadra and Mustafa E. Şeker and Mustafa S. Kartal and Noëlie Debs and Richard E. Fan and Sara Saunders and Simon J.C. Soerensen and Stefania Moroianu and Sulaiman Vesal and Yuan Yuan and Afsoun Malakoti-Fard and Agnė Mačiūnien and Akira Kawashima and Ana M.M. de M.G. {de Sousa Machadov} and Ana Sofia L. Moreira and Andrea Ponsiglione and Annelies Rappaport and Arnaldo Stanzione and Arturas Ciuvasovas and Baris Turkbey and Bart {de Keyzer} and Bodil G. Pedersen and Bram Eijlers and Christine Chen and Ciabattoni Riccardo and Deniz Alis and Ewout F.W. {Courrech Staal} and Fredrik Jäderling and Fredrik Langkilde and Giacomo Aringhieri and Giorgio Brembilla and Hannah Son and Hans Vanderlelij and Henricus P.J. Raat and Ingrida Pikūnienė and Iva Macova and Ivo Schoots and Iztok Caglic and Jeries P. Zawaideh and Jonas Wallström and Leonardo K. Bittencourt and Misbah Khurram and Moon H. Choi and Naoki Takahashi and Nelly Tan and Paolo N. Franco and Patricia A. Gutierrez and Per Erik Thimansson and Pieter Hanus and Philippe Puech and Philipp R. Rau and Pieter {de Visschere} and Ramette Guillaume and Renato Cuocolo and Ricardo O. Falcão and Rogier S.A. {van Stiphout} and Rossano Girometti and Ruta Briediene and Rūta Grigienė and Samuel Gitau and Samuel Withey and Sangeet Ghai and Tobias Penzkofer and Tristan Barrett and Varaha S. Tammisetti and Vibeke B. Løgager and Vladimír Černý and Wulphert Venderink and Yan M. Law and Young J. Lee and Maarten {de Rooij} and Henkjan Huisman},
}
@article{Bosma23,
    author={Joeran S. Bosma, Anindo Saha, Matin Hosseinzadeh, Ivan Slootweg, Maarten de Rooij, and Henkjan Huisman},
    title={Semisupervised Learning with Report-guided Pseudo Labels for Deep Learning–based Prostate Cancer Detection Using Biparametric MRI},
    journal={Radiology: Artificial Intelligence},
    pages={e230031},
    year={2023},
    doi={10.1148/ryai.230031},
    publisher={Radiological Society of North America}
}
```

## Managed By
Diagnostic Image Analysis Group,
Radboud University Medical Center,
Nijmegen, The Netherlands

## Contact Information
- Joeran Bosma: Joeran.Bosma@radboudumc.nl
- Anindo Saha: Anindya.Shaha@radboudumc.nl
- Henkjan Huisman: Henkjan.Huisman@radboudumc.nl

[picai_nnunet_docker]: https://hub.docker.com/r/joeranbosma/picai_nnunet
[picai_nndetection_docker]: https://hub.docker.com/r/joeranbosma/picai_nndetection
[picai_prep]: https://github.com/DIAGNijmegen/picai_prep
[nnunet_raw_data_format]: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md
[picai_prep_mha2nnunet]: https://github.com/DIAGNijmegen/picai_prep#mha-archive--nnu-net-raw-data-archive
[nnunet-archive]: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md
[nndetection-archive]: https://github.com/MIC-DKFZ/nnDetection/#adding-new-data-sets
