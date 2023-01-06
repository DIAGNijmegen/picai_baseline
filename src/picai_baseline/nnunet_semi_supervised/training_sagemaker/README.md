# PI-CAI Closed Test Phase Submission

## 1. Docker container
Define the Docker container that can be used to preprocess the data, and train the models. The Docker build process must run on Linux, including the systems from the PI-CAI organizers, to ensure reproducibility. For this example, we will use the [PI-CAI nnU-Net Docker container](https://github.com/DIAGNijmegen/picai_baseline/tree/main/src/picai_baseline/nnunet/training_docker). You can [install any public library](https://github.com/DIAGNijmegen/picai_baseline/blob/2ad6b5aa03a22633ef2cbecfeeefc1efe6f9b01a/src/picai_baseline/nnunet/training_docker/Dockerfile#L36), and [add custom code to the Docker environment](https://github.com/DIAGNijmegen/picai_baseline/blob/2ad6b5aa03a22633ef2cbecfeeefc1efe6f9b01a/src/picai_baseline/nnunet/training_docker/Dockerfile#L38-L53).


## 2. Preprocessing
This script is used to preprocess the dataset, in our example for training within the nnU-Net framework. The script expects to be passed a few arguments, including the directories where the images and labels can be found.

The script extracts custom code (from `code.zip`) for usage during preprocessing, which contains the [PI-CAI semi-supervised data preprocessing script](https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/prepare_data_semi_supervised.py) in this example.

The script then loads the cross-validation splits for the dataset, either from a JSON file specified by the splits argument, or from one of several predefined split configurations specified by the splits argument. To debug this script, you can use the public PI-CAI Training and Development dataset (`--splits=picai_pub`), which can be changed to the PI-CAI Public and Private Training dataset at submission (`--splits=picai_pubpriv`).

The provided preprocessing script follows the preprocessing steps outlined in the [PI-CAI Baseline tutorial](https://github.com/DIAGNijmegen/picai_baseline), plus the conversion to the nnU-Net preprocessed format.

At the end of the script, the preprocessed dataset is exported by copying it to the directory specified by `args.outputdir`. Please note, that all other resources are destroyed! As such, supporting files like `dataset.json` must be exported too!

Note: no training happens yet. Ideally, this step should not require a GPU.


## 3. Training
This script is for training your method, with the provided example training a semi-supervised nnU-Net model. The model is trained on a dataset of images and corresponding labels (segmentation masks), which were preprocessed in step 2. Preprocessing. The script takes in several arguments including the input image and label directories, output directory, and number of folds to train the model on.

The script first defines a function `main()` that sets up an argument parser to parse the input arguments. Then it sets up various paths and directories based on the input arguments and environment variables. It also extracts code from a zip file located in the scripts directory. You can provide your custom scripts in this `code.zip` file.

The train script trains the model on the specified folds by calling the 'nnunet' command, which is defined through the setup in the definition of the Docker container. 

Finally, all necessary resources for inference are exported to the directory specified by `args.outputdir`. Please note, that all other resources are destroyed! As such, supporting files for inference (like `plans.pkl` for nnU-Net) must be exported too!


## Folder structure
During preprocessing and training, the Docker container will be run on AWS SageMaker. In this environment, multiple storage types exist:

1. There is about 15 GB of available disk space for local processes (e.g., at `/opt/algorithm`)
2. A user-configurable amount of temporary storage (removed after preprocessing/training) which is mounted on `/tmp`
3. Input folders (e.g., with the PI-CAI images or labels) that are mounted at the location specified by `args.imagesdir`, `args.labelsdir`.
4. Output folder (e.g., to store the preprocessed dataset or trained model resources), mounted at the location specified by `args.outputdir`.

The PI-CAI Public and Private Training dataset is approx. 150 GB, so any preprocessing step likely must be performed within the temporary storage at `/tmp`. For the nnU-Net preprocessing pipeline, the provided MHA Archive is first converted to the [nnU-Net Raw Data Archive](https://github.com/DIAGNijmegen/picai_prep#mha-archive--nnu-net-raw-data-archive) (i.e., + 150 GB of storage required), after which nnU-Net crops the images (i.e., + 185 GB of storage required*), and preprocessed the images (i.e., + 450 GB of storage required*). As such, the temporary disk space requirement is high. At the end of the preprocessing script, the preprocessed images are exported. This saves the disk space for 150+185=235 GB worth of images.

*nnU-Net's formats are less efficient than the raw images.