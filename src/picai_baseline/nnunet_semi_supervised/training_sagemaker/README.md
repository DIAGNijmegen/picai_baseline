# PI-CAI Closed Test Phase Submission

## 1. Docker container
Define the Docker container that can be used to preprocess the data, and train the models. The Docker build process must run on Linux, including the systems from the PI-CAI organizers, to ensure reproducibility. For this example, we will use the [PI-CAI nnU-Net Docker container](https://github.com/DIAGNijmegen/picai_baseline/tree/main/src/picai_baseline/nnunet/training_docker). You can [install any public library](https://github.com/DIAGNijmegen/picai_baseline/blob/2ad6b5aa03a22633ef2cbecfeeefc1efe6f9b01a/src/picai_baseline/nnunet/training_docker/Dockerfile#L36), and [add custom code to the Docker environment](https://github.com/DIAGNijmegen/picai_baseline/blob/2ad6b5aa03a22633ef2cbecfeeefc1efe6f9b01a/src/picai_baseline/nnunet/training_docker/Dockerfile#L38-L53).


## 2. Preprocessing
This script is used to preprocess the dataset, in our example for training within the nnU-Net framework. The script expects to be passed a few arguments, including the directories where the images and labels can be found.

This script loads the cross-validation splits for the dataset, either from a JSON file specified by the splits argument, or from one of several predefined split configurations specified by the splits argument. To debug this script, you can use the public PI-CAI Training and Development dataset (`--splits=picai_pub`), which can be changed to the PI-CAI Public and Private Training dataset at submission (`--splits=picai_pubpriv`).

The provided preprocessing script follows the preprocessing steps outlined in the [PI-CAI Baseline tutorial](https://github.com/DIAGNijmegen/picai_baseline), plus the conversion to the nnU-Net preprocessed format.

At the end of the script, the preprocessed dataset is exported by copying it to the directory specified by `args.outputdir`. Please note, that all other resources are destroyed! As such, supporting files like `dataset.json` must be exported too!

Note: no training happens yet. Ideally, this step should not require a GPU.

The preprocessing pipeline can be tested locally by adapting the [`run_preprocessing_debug.sh`](run_preprocessing_debug.sh) or [`run_preprocessing_public.sh`](run_preprocessing_public.sh) script. To do this, set up the paths to the required input and output folders, which are detailed below.


## 3. Training
This script is for training your method, with the provided example training a semi-supervised nnU-Net model. The model is trained on a dataset of images and corresponding labels (segmentation masks), which were preprocessed in step 2. Preprocessing. The script takes in several arguments including the input image and label directories, output directory, and number of folds to train the model on.

The script first defines a function `main()` that sets up an argument parser to parse the input arguments. Then it sets up various paths and directories based on the input arguments and environment variables.

The train script trains the model on the specified folds by calling the `nnunet` command, which is defined through the setup in the definition of the Docker container. 

Finally, all necessary resources for inference are exported to the directory specified by `args.outputdir`. Please note, that all other resources are destroyed! As such, supporting files for inference (like `plans.pkl` for nnU-Net) must be exported too!

Each training session may run for 5 days at most. As this is typically insufficient to train all components, the training pipeline must be divided into chunks. For the nnU-Net baseline in this example, we made the fold configurable. Each fold takes ~2 days to train, so we can split up training in appropriate chunks.

The training pipeline can be tested locally by adapting the [`run_training_local.sh`](run_training_local.sh) script. To do this, set up the paths to the required input and output folders, which are detailed below.


## Folder structure
During preprocessing and training, the Docker container will be run on AWS SageMaker. In this environment, multiple storage types exist:

1. There is about 15 GB of available disk space for local processes (e.g., at `/opt/algorithm`)
2. A user-configurable amount of temporary storage (removed after preprocessing/training) which is mounted on `/tmp`
3. Input folders (e.g., with the PI-CAI images or labels) that are mounted at the location specified by `args.imagesdir`, `args.labelsdir`.
4. Output folder (e.g., to store the preprocessed dataset or trained model resources), mounted at the location specified by `args.outputdir`.

The PI-CAI Public and Private Training dataset is approx. 150 GB, so any preprocessing step likely must be performed within the temporary storage at `/tmp`. For the nnU-Net preprocessing pipeline, the provided MHA Archive is first converted to the [nnU-Net Raw Data Archive](https://github.com/DIAGNijmegen/picai_prep#mha-archive--nnu-net-raw-data-archive) (i.e., + 150 GB of storage required), after which nnU-Net crops the images (i.e., + 185 GB of storage required*), and preprocessed the images (i.e., + 450 GB of storage required*). As such, the temporary disk space requirement is high. At the end of the preprocessing script, the preprocessed images are exported. This saves the disk space for 150+185=235 GB worth of images.

*nnU-Net's formats are less efficient than the raw images.

## Input/output folders
During **preprocessing**, the following folders will be available:

1. MHA Archive, located at `args.imagesdir`. This folder contains the imaging dataset in the [MHA Archive format](https://github.com/DIAGNijmegen/picai_prep#what-is-an-mha-archive). The folder structure will be the same as during the Open Development phase (see [here]((https://github.com/DIAGNijmegen/picai_prep#what-is-an-mha-archive))). To test your pipeline locally, mount this folder read-only: `-v /path/to/images:/input/images:ro`.
2. Labels, located at `args.labelsdir`. This folder contains all annotations: case-level marksheet, csPCa lesion delineations (manual and AI-derived), and prostate segmentations. The folder structure will be the same as [`picai_labels`](https://github.com/DIAGNijmegen/picai_labels). To test your pipeline locally, mount this folder read-only: `-v /path/to/picai_labels:/input/picai_labels:ro`.
3. Output folder, located at `args.outputdir`. This folder must contain the preprocessed dataset at the end of your preprocessing pipeline. Any file that's not in this folder, will be lost! To test your pipeline locally, mount this folder: `-v /path/to/preprocessed:/output`.

Optionally, you can mount a folder to `/workdir` to debug/see what's going on in your pipeline: `-v /path/to/workdir:/workdir`. Depending on your setup, you can mount a folder with the preprocessing pipeline itself: `-v /path/to/code:/code:ro`.

-------

During **training**, the following folders will be available:

1. MHA Archive, see above.
2. Labels, see above.
3. Preprocessed dataset, located at `args.preprocesseddir`. This folder contains the output of your preprocessing pipeline. To test your pipeline locally, mount this folder read-only: `-v /path/to/preprocessed:/input/preprocessed:ro`.
4. Model checkpoints, located at `args.checkpointsdir`. (Optional) This folder can be used to store model checkpoints during training, to ensure training is set up correctly. In the event training fails, the checkpoints directory can be used to resume model training. We encourage the usage of this, as it makes the training pipeline more robust, and do the same in our reference implementation. To test your pipeline locally, mount this folder: `-v /path/to/checkpoints:/checkpoints`.
5. Output folder, located at `args.outputdir`. This folder must contain the trained algorithm checkpoints for inference, and any supporting file required to perform inference. Any file that's not in this folder, will be lost! To test your pipeline locally, mount this folder: `-v /path/to/preprocessed:/output`.

Optionally, you can mount a folder to `/workdir` to debug/see what's going on in your pipeline: `-v /path/to/workdir:/workdir`. Depending on your setup, you can mount a folder with the preprocessing pipeline itself: `-v /path/to/code:/code:ro`.

-------

Prostate segmentations will be made available to all teams. We provide whole-gland prostate segmentations from [this algorithm](https://grand-challenge.org/algorithms/prostate-segmentation/) at `/input/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b` (i.e., same as for the PI-CAI Open Development Phase).

Custom (whole-gland/zonal) prostate segmentation masks from teams participating in the Closed Testing Phase will be made available in the same format. We will make segmentation masks available with these methods for the public dataset too, so the exact folder structure will be visible in [`picai_labels`](https://github.com/DIAGNijmegen/picai_labels/tree/main/anatomical_delineations). Segmentation masks for the PI-CAI Private Training dataset will be available to any team participating in the Closed Testing Phase.
