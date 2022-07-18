[← Return to overview](https://github.com/DIAGNijmegen/picai_baseline#baseline-ai-models-for-prostate-cancer-detection-in-mri)

#

## nnDetection
The nnDetection framework is geared towards medical object detection [[2]](https://github.com/DIAGNijmegen/picai_baseline#2). In its native form, nnDetection will predict bounding boxes, which may overlap. The PI-CAI challenge requires detection maps of non-touching lesion candidates, so we will transform the bounding box predictions to detection maps after inference with nnDetection. Setting up nnDetection and tweaking its implementation is not as straightforward as for the [nnUNet](nnunet_baseline.md) or [UNet](unet_baseline.md) baselines, but it can provide a strong csPCa detection model. Interested readers who would like to modify the implementation of nnDetection are referred to the [nnDetection documentation](https://github.com/MIC-DKFZ/nnDetection/). We only provide training and evaluation steps with the vanilla nnDetection framework.


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

We advice to export the cross-validation splits as individual files for usage with `picai_eval`. To achieve this, please follow the steps in [Cross-Validation Splits](https://github.com/DIAGNijmegen/picai_baseline#cross-validation-splits), or run this with Docker:

```bash
docker run --cpus=1 --memory=4gb -it --rm \
    -v /path/to/workdir:/workdir \
    joeranbosma/picai_nndetection:latest \
    python -m picai_baseline.splits.picai_nnunet --output "/workdir/splits/picai_nnunet"
```


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


### nnDetection - Inference
Before inference with nnDetection, consolidate the models first. See [nnDetection's documentation](https://github.com/MIC-DKFZ/nnDetection/#inference) for details. With the `picai_nndet` Docker container, models can be consolidated using the following command:

```bash
docker run --cpus=8 --memory=32gb --shm-size=32gb --gpus='"device=0"' -it --rm \
    -v /path/to/workdir:/workdir \
    joeranbosma/picai_nndetection:latest nndet consolidate \
    Task2201_picai_baseline RetinaUNetV001_D3V001_3d /workdir \
    --sweep_boxes \
    --results /workdir/results/nnDet \
    --custom_split /workdir/nnUNet_raw_data/Task2201_picai_baseline/splits.json
```

This will generate an inference plan for model deployment. Additionally, this will generate cross-validation predictions of bounding boxes, which can be used for internal model development. See [nnDetection - Evaluation](#nndetection---evaluation) for instruction on how to evaluate these predictions in the context of the PI-CAI Challenge.

To predict unseen images with the consolidated nnDetection models (i.e., cross-validation ensemble) you can use the following command:

```bash
docker run --cpus=8 --memory=32gb --shm-size=32gb --gpus='"device=0"' -it --rm \
    -v /path/to/workdir:/workdir \
    -v /path/to/images:/input/images \
    joeranbosma/picai_nndetection:latest nndet predict Task2201_picai_baseline RetinaUNetV001_D3V001_3d /workdir \
    --fold -1 --check --resume --input /input/images --output /workdir/predictions/ --results /workdir/results/nnDet
```


### nnDetection - Evaluation
For cross-validation with predictions from [`nndet consolidate`](#nndetection---inference), generate detection maps for each fold. We provide a simple script for this, which transforms bounding boxes into cubes with the corresponding lesion confidence. All bounding boxes that overlap with another bounding box of higher confidence are discarded, to conform with the non-touching lesion candidates required by the PI-CAI Challenge.

Note: this is by no means the best strategy to transform bounding boxes to detection maps. We leave it to participants to improve on this translation step, e.g. by using spheres instead of cubes.

To convert boxes to detection maps with Docker:

```bash
docker run --cpus=4 --memory=32gb --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nndetection:latest python /opt/code/nndet_generate_detection_maps.py --input /workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/fold0/val_predictions
docker run --cpus=4 --memory=32gb --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nndetection:latest python /opt/code/nndet_generate_detection_maps.py --input /workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/fold1/val_predictions
docker run --cpus=4 --memory=32gb --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nndetection:latest python /opt/code/nndet_generate_detection_maps.py --input /workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/fold2/val_predictions
docker run --cpus=4 --memory=32gb --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nndetection:latest python /opt/code/nndet_generate_detection_maps.py --input /workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/fold3/val_predictions
docker run --cpus=4 --memory=32gb --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nndetection:latest python /opt/code/nndet_generate_detection_maps.py --input /workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/fold4/val_predictions
```

To evaluate, we can use the `picai_eval` repository, see [here](https://github.com/DIAGNijmegen/picai_eval) for documentation. For evaluation from Python, you can adapt the script given in [nnU-Net - Evaluation](nnunet_baseline.md#nnu-net---evaluation) (you can remove the lesion exaction, but this is not necessary, as lesions extraction from a detection map gives the detection map itself).

For evaluation from the command line/Docker, we advice to [save the subject lists to disk](https://github.com/DIAGNijmegen/picai_baseline#cross-validation-splits) first, so you can ensure no cases are missing. Then, you can evaluate the detection maps using the following commands:

```bash
docker run --cpus=4 --memory=32gb --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nndetection:latest python -m picai_eval --input /workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/fold0/val_predictions_detection_maps --labels /workdir/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/ --subject_list /workdir/splits/picai_nnunet/ds-config-valid-fold-0.json
docker run --cpus=4 --memory=32gb --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nndetection:latest python -m picai_eval --input /workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/fold1/val_predictions_detection_maps --labels /workdir/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/ --subject_list /workdir/splits/picai_nnunet/ds-config-valid-fold-1.json
docker run --cpus=4 --memory=32gb --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nndetection:latest python -m picai_eval --input /workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/fold2/val_predictions_detection_maps --labels /workdir/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/ --subject_list /workdir/splits/picai_nnunet/ds-config-valid-fold-2.json
docker run --cpus=4 --memory=32gb --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nndetection:latest python -m picai_eval --input /workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/fold3/val_predictions_detection_maps --labels /workdir/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/ --subject_list /workdir/splits/picai_nnunet/ds-config-valid-fold-3.json
docker run --cpus=4 --memory=32gb --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nndetection:latest python -m picai_eval --input /workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/fold4/val_predictions_detection_maps --labels /workdir/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/ --subject_list /workdir/splits/picai_nnunet/ds-config-valid-fold-4.json
```

The metrics will be displayed in the command line and stored to `metrics.json` (inside the `--input` directory). To load the metrics for subsequent analysis, we recommend loading the metrics using `picai_eval`, this allows on-the-fly calculation of metrics (described in more detail [here](https://github.com/DIAGNijmegen/picai_eval#accessing-metrics-after-evaluation)).


### nnDetection - Algorithm Submission
Once training is complete, you are ready to make an algorithm submission. Please read about [Submission of Inference Containers to the Open Development Phase](https://pi-cai.grand-challenge.org/ai-algorithm-submissions/) first. The grand-challenge algorithm submission template for this algorithm can be found [here](https://github.com/DIAGNijmegen/picai_nndetection_gc_algorithm).

To deploy your own nnDetection algorithm, the trained models need to be transferred. Inference with nnDetection requires the following files (for the task name and trainer specified above):

```bash
~/workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated
├── config.yaml
├── model_fold0.ckpt
├── model_fold1.ckpt
├── model_fold2.ckpt
├── model_fold3.ckpt
├── model_fold4.ckpt
└── plan_inference.pkl
```

As well as:

```bash
~/workdir/nnDet_raw_data/Task2201_picai_baseline/dataset.json
```


These files can be collected through the command-line as follows:

```bash
mkdir -p /path/to/repos/picai_nndetection_gc_algorithm/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated
cp /path/to/workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/config.yaml /path/to/repos/picai_nndetection_gc_algorithm/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/config.yaml
cp /path/to/workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/model_fold0.ckpt /path/to/repos/picai_nndetection_gc_algorithm/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/model_fold0.ckpt
cp /path/to/workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/model_fold1.ckpt /path/to/repos/picai_nndetection_gc_algorithm/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/model_fold1.ckpt
cp /path/to/workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/model_fold2.ckpt /path/to/repos/picai_nndetection_gc_algorithm/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/model_fold2.ckpt
cp /path/to/workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/model_fold3.ckpt /path/to/repos/picai_nndetection_gc_algorithm/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/model_fold3.ckpt
cp /path/to/workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/model_fold4.ckpt /path/to/repos/picai_nndetection_gc_algorithm/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/model_fold4.ckpt
cp /path/to/workdir/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/plan_inference.pkl /path/to/repos/picai_nndetection_gc_algorithm/results/nnDet/Task2201_picai_baseline/RetinaUNetV001_D3V001_3d/consolidated/plan_inference.pkl
cp /path/to/workdir/nnDet_raw_data/Task2201_picai_baseline/dataset.json /path/to/repos/picai_nndetection_gc_algorithm/results/nnDet/Task2201_picai_baseline/dataset.json
```

After collecting these files, please continue with the instructions provided in [Submission of Inference Containers to the Open Development Phase](https://pi-cai.grand-challenge.org/ai-algorithm-submissions/).


### nnDetection - Semi-supervised Training
The [semi-supervised nnDetection model](https://github.com/DIAGNijmegen/picai_nndetection_semi_supervised_gc_algorithm) is trained in a very similar manner as the supervised nnDetection model. To train the semi-supervised model, prepare the dataset using [`prepare_data_semi_supervised.py`](src/picai_baseline/prepare_data_semi_supervised.py). See [Data Preprocessing](README.md#data-preprocessing) for details. Then, follow the steps above, replacing `Task2201_picai_baseline` with `Task2203_picai_baseline`.


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
