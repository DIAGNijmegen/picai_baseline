[← Return to overview](https://github.com/DIAGNijmegen/picai_baseline#baseline-ai-models-for-prostate-cancer-detection-in-mri)

#

## nnU-Net
The nnU-Net framework [[1]](#1) provides a performant framework for medical image segmentation, which is straightforward to adapt for csPCa detection. To use nnUNet for detection, it is important to replace the default Cross-Entropy + soft Dice loss [[2]](#2)[[3]](#3). Good options for the loss function are, e.g., Cross-Entropy, or Cross-Entropy + Focal loss [[4]](#4).

We provide all steps to train nnU-Net in [nnunet_baseline.py](src/picai_baseline/nnunet/nnunet_baseline.py), and provide extended documentation below.


### nnU-Net - Docker Setup
To run nnU-Net commands, you can use the Docker specified in [`nnunet/training_docker/`](src/picai_baseline/nnunet/training_docker/). This is a wrapper around nnUNet, and facilitates training in a Docker container on a distributed system. Additionally, this Docker container shows how custom nnU-Net trainers can be implemented (shown for [`nnUNetTrainerV2_Loss_FL_and_CE`](src/picai_baseline/nnunet/training_docker/nnUNetTrainerV2_Loss_FL_and_CE.py)).

To build the Docker container, navigate to [`nnunet/training_docker/`](src/picai_baseline/nnunet/training_docker/) and build the container:

```
cd src/picai_baseline/nnunet/training_docker/
docker build . --tag joeranbosma/picai_nnunet:latest
```

This will result (if ran successfully) in the Docker container named `joeranbosma/picai_nnunet:latest`. Alternatively, the pre-built Docker container can be loaded:

```
docker pull joeranbosma/picai_nnunet:latest
```


### nnU-Net - Cross-Validation Splits
nnU-Net can train with user-defined cross-validation splits, which is required to ensure there is no patient overlap between training and validation splits. The [cross-validation splits](#cross-validation splits) can be copied to nnU-Net's working directory as follows:

```python
from picai_baseline.splits.picai_nnunet import nnunet_splits

nnUNet_splits_path = "/workdir/nnUNet_raw_data/Task2201_picai_baseline/splits.json"

# save cross-validation splits to disk
with open(nnUNet_splits_path, "w") as fp:
    json.dump(nnunet_splits, fp)
```

Alternatively, the generated [splits.json](src/picai_baseline/splits/picai_nnunet/splits.json) can be copied directly.


### nnU-Net - Training
For general documentation on how to train nnUNet models, please check the [official documentation](https://github.com/MIC-DKFZ/nnUNet#usage). We use a wrapper around nnUNet to orchestrate the `nnUNet_plan_and_preprocess` and `nnUNet_train` steps. Our baseline model swaps out Cross-Entropy + soft Dice loss for Cross-Entropy + Focal loss.

Running the first fold will start with preprocessing the raw images. After preprocessing is done, it will automatically start training.

Note: the provided baseline uses Cross-Entropy + Focal loss (`nnUNetTrainerV2_Loss_FL_and_CE_checkpoints` trainer), as defined in the [`picai_nnunet`](src/picai_baseline/nnunet/training_docker) Docker. You can also use the `nnUNetTrainerV2_Loss_CE` trainer for Cross-Entropy loss, which is available with the official nnU-Net installation.

```bash
docker run --cpus=8 --memory=64gb --shm-size=64gb --gpus='"device=0"' --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nnunet:latest nnunet plan_train Task2201_picai_baseline /workdir/ --trainer nnUNetTrainerV2_Loss_FL_and_CE_checkpoints --fold 0 --custom_split /workdir/nnUNet_raw_data/Task2201_picai_baseline/splits.json
docker run --cpus=8 --memory=64gb --shm-size=64gb --gpus='"device=1"' --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nnunet:latest nnunet plan_train Task2201_picai_baseline /workdir/ --trainer nnUNetTrainerV2_Loss_FL_and_CE_checkpoints --fold 1 --custom_split /workdir/nnUNet_raw_data/Task2201_picai_baseline/splits.json
docker run --cpus=8 --memory=64gb --shm-size=64gb --gpus='"device=2"' --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nnunet:latest nnunet plan_train Task2201_picai_baseline /workdir/ --trainer nnUNetTrainerV2_Loss_FL_and_CE_checkpoints --fold 2 --custom_split /workdir/nnUNet_raw_data/Task2201_picai_baseline/splits.json
docker run --cpus=8 --memory=64gb --shm-size=64gb --gpus='"device=3"' --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nnunet:latest nnunet plan_train Task2201_picai_baseline /workdir/ --trainer nnUNetTrainerV2_Loss_FL_and_CE_checkpoints --fold 3 --custom_split /workdir/nnUNet_raw_data/Task2201_picai_baseline/splits.json
docker run --cpus=8 --memory=64gb --shm-size=64gb --gpus='"device=4"' --rm -v /path/to/workdir:/workdir/ joeranbosma/picai_nnunet:latest nnunet plan_train Task2201_picai_baseline /workdir/ --trainer nnUNetTrainerV2_Loss_FL_and_CE_checkpoints --fold 4 --custom_split /workdir/nnUNet_raw_data/Task2201_picai_baseline/splits.json

```

After preprocessing is done, the other folds can be run sequentially or in parallel with the first fold (change the `--gpus` flag accordingly).

Note: runs in our environment with 28 GB RAM, 8 CPUs, 1 GPU with 8 GB VRAM. Takes about 2-3 days per fold on an RTX 2080 Ti.


### nnU-Net - Inference
After training nnU-Net to convergence (i.e., after 1000 epochs), we can perform inference using nnUNet's 'best' model, `model_best`, or the final checkpoint, `model_final`. We can use a single model for cross-validation, or ensemble the models for the test set.

To evaluate individual models for cross-validation:

```bash
docker run --cpus=8 --memory=28gb --gpus='"device=0"' --rm \
    -v /path/to/test_set/images:/input/images \
    -v /path/to/workdir/results:/workdir/results \
    -v /path/to/workdir/predictions:/output/predictions \
    joeranbosma/picai_nnunet:latest nnunet predict Task2201_picai_baseline \
    --trainer nnUNetTrainerV2_Loss_FL_and_CE_checkpoints \
    --fold 0 --checkpoint model_best \
    --results /workdir/results \
    --input /input/images/ \
    --output /output/predictions \
    --store_probability_maps
```

Repeat this for each fold (change `validation_set/fold_1`, `predictions_fold_1` and `--fold 1`, etc.)


### nnU-Net - Evaluation
To evaluate, we can use the `picai_eval` repository, see [here](https://github.com/DIAGNijmegen/picai_eval) for documentation.
The nnU-Net framework generates _softmax predictions_, while we need _detection maps_ for the PI-CAI challenge. To extract lesion candidates from softmax predictions, we can use the [Report-Guided Annotation repository](https://github.com/DIAGNijmegen/Report-Guided-Annotation). This repository has a [dynamic lesion extraction method](https://github.com/DIAGNijmegen/Report-Guided-Annotation/blob/main/src/report_guided_annotation/extract_lesion_candidates.py), described in [[3]](#3). The lesion candidates from this method are (by design) compatible with the PI-CAI evaluation pipeline. The Report-Guided Annotation repository should be automatically installed when installing the `picai_baseline` repository, and is installed within the [`picai_nnunet`][picai_nnunet_docker] and [`picai_nndetection`][picai_nndetection_docker] Docker containers as well.

The nnU-Net softmax predictions (saved as .npz files) cannot be used directly, because these pertain to the cropped/preprocessed images, instead of the original images. To solve this, we wrote a helper function in [picai_baseline/nnunet/softmax_export.py](src/picai_baseline/nnunet/softmax_export.py) to convert a cropped prediction to its original extent.

All of the above steps are combined in [picai_baseline/nnunet/eval.py](src/picai_baseline/nnunet/eval.py), enabling evaluation from the command line:

```bash
python /path/to/picai_baseline/nnunet/eval.py --task=Task2201_picai_baseline --workdir=/path/to/workdir
```

Or from Python:

```python
from picai_baseline.nnunet.eval import evaluate

# evaluate
evaluate(
    task="Task2201_picai_baseline",
    workdir="/path/to/workdir",
)
```

Or with Docker:

```bash
docker run --cpus=4 --memory=16gb --rm \
    -v /path/to/workdir/:/workdir/ \
    -v /path/to/repos/:/repos/ \
    joeranbosma/picai_nnunet:latest python3 /repos/picai_baseline/src/picai_baseline/nnunet/eval.py --task=Task2201_picai_baseline
```

The metrics will be displayed in the command line and stored to `metrics-{checkpoint}-{threshold}.json` (by default in `/path/to/workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_[0,1,2,3,4]`). To see additional options and default parameters, please refer to the command line help (`python src/picai_baseline/nnunet/eval.py -h`) or the [source code](src/picai_baseline/nnunet/eval.py).

To load the metrics for subsequent analysis, we recommend loading the metrics using `picai_eval`, this allows on-the-fly calculation of metrics (described in more detail [here](https://github.com/DIAGNijmegen/picai_eval#accessing-metrics-after-evaluation)):

```python
from picai_eval import Metrics

fold = 0
checkpoint = "model_best"
threshold = "dynamic"
metrics = Metrics(f"/path/to/workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_{fold}/metrics-{checkpoint}-{threshold}.json")
print(f"PI-CAI ranking score: {metrics.score:.4f} " +
      + f"(50% AUROC={metrics.auroc:.4f} + 50% AP={metrics.AP:.4f})")
```

### nnU-Net - Algorithm Submission to Grand Challenge
Once training is complete, you are ready to make an algorithm submission. Please read about [Submission of Inference Containers to the Open Development Phase](https://pi-cai.grand-challenge.org/ai-algorithm-submissions/) first. The grand-challenge algorithm submission template for this algorithm can be found [here](https://github.com/DIAGNijmegen/picai_nnunet_gc_algorithm).

To deploy your own nnU-Net algorithm, the trained models need to be transferred. Inference with nnU-Net requires the following files (for the task name and trainer specified above):

```bash
~/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1
├── fold_0
│   ├── model_best.model
│   └── model_best.model.pkl
├── fold_1/...
├── fold_2/...
├── fold_3/...
├── fold_4/...
└── plans.pkl
```

These files can be collected through Python (see [`nnunet_baseline.py`](src/picai_baseline/nnunet/nnunet_baseline.py)), or through the command-line:

```bash
cp /workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_0/model_best.model /path/to/repos/picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_0/model_best.model
cp /workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_0/model_best.model.pkl /path/to/repos/picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_0/model_best.model.pkl
cp /workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_1/model_best.model /path/to/repos/picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_1/model_best.model
cp /workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_1/model_best.model.pkl /path/to/repos/picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_1/model_best.model.pkl
cp /workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_2/model_best.model /path/to/repos/picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_2/model_best.model
cp /workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_2/model_best.model.pkl /path/to/repos/picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_2/model_best.model.pkl
cp /workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_3/model_best.model /path/to/repos/picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_3/model_best.model
cp /workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_3/model_best.model.pkl /path/to/repos/picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_3/model_best.model.pkl
cp /workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_4/model_best.model /path/to/repos/picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_4/model_best.model
cp /workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_4/model_best.model.pkl /path/to/repos/picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_4/model_best.model.pkl
cp /workdir/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/plans.pkl /path/to/repos/picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/Task2201_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/plans.pkl
```

After collecting these files, please continue with the instructions provided in [Submission of Inference Containers to the Open Development Phase](https://pi-cai.grand-challenge.org/ai-algorithm-submissions/).


### nnU-Net - Semi-supervised Training
The [semi-supervised nnU-Net model](https://github.com/DIAGNijmegen/picai_nnunet_semi_supervised_gc_algorithm) is trained in a very similar manner as the supervised nnU-Net model. To train the semi-supervised model, prepare the dataset using [`prepare_data_semi_supervised.py`](src/picai_baseline/prepare_data_semi_supervised.py). See [Data Preprocessing](README.md#data-preprocessing) for details. Then, follow the steps above, replacing `Task2201_picai_baseline` with `Task2203_picai_baseline`.


## References
<a id="1" href="https://www.nature.com/articles/s41592-020-01008-z">[1]</a> 
Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen and Klaus H. Maier-Hein. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation". Nature Methods 18.2 (2021): 203-211.

<a id="2" href="https://link.springer.com/chapter/10.1007/978-3-030-87240-3_51">[2]</a> 
Michael Baumgartner, Paul F. Jaeger, Fabian Isensee, Klaus H. Maier-Hein. "nnDetection: A Self-configuring Method for Medical Object Detection". International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2021.

<a id="3" href="https://arxiv.org/abs/2112.05151">[3]</a> 
Joeran Bosma, Anindo Saha, Matin Hosseinzadeh, Ilse Slootweg, Maarten de Rooij, Henkjan Huisman. "Semi-supervised learning with report-guided lesion annotation for deep learning-based prostate cancer detection in bpMRI". arXiv:2112.05151.

<a id="4" href="#">[4]</a> 
Joeran Bosma, Natalia Alves and Henkjan Huisman. "Performant and Reproducible Deep Learning-Based Cancer Detection Models for Medical Imaging". _Under Review_.


[picai_nnunet_docker]: https://hub.docker.com/r/joeranbosma/picai_nnunet
[picai_nndetection_docker]: https://hub.docker.com/r/joeranbosma/picai_nndetection
