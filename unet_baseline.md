[← Return to overview](https://github.com/DIAGNijmegen/picai_baseline#baseline-ai-models-for-prostate-cancer-detection-in-mri)

#

## U-Net
We include a lightweight, baseline [U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) to provide a playground environment for participants and kickstart their development cycle. Its goal is to help developers get familiar with the end-to-end pipeline of training a U-Net model for csPCa detection/diagnosis in 3D, encapsulating the trained AI model in a Docker container, and uploading the same to the [grand-challenge.org](https://grand-challenge.org/) platform as an ["algorithm"](https://grand-challenge.org/documentation/algorithms/). The U-Net baseline generates quick results with minimal complexity, but does so at the expense of sub-optimal performance and low flexibility in adapting to any other task.


### U-Net - Data Preparation
We use the [same cross-validation splits](README.md#cross-validation-splits) for this U-Net, as the [nnU-Net](nnunet_baseline.md). We use the same data preparation/preprocessing pipeline for this U-Net, as the [nnU-Net](nnunet_baseline.md), with two exceptions. We specify the following in [Line 84-85 of prepare_data.py](src/picai_baseline/prepare_data.py#L84-L85):

```python
mha2nnunet_settings["preprocessing"]["spacing"] = [3.0, 0.5, 0.5]
mha2nnunet_settings["preprocessing"]["matrix_size"] = [20, 256, 256]
```

Instead of nnUNet's:

```python
mha2nnunet_settings["preprocessing"]["physical_size"] = [81.0, 192.0, 192.0]
mha2nnunet_settings["preprocessing"]["crop_only"] = True
```

By doing so, besides formatting and converting the dataset into the [`nnU-Net Raw Data Archive`][nnunet_raw_data_format] structure, [prepare_data.py](src/picai_baseline/prepare_data.py) also preprocesses each scan and annotation in the dataset, as follows:

- **Resampling Spatial Resolution**: The [PI-CAI: Public Training and Development Dataset](https://pi-cai.grand-challenge.org/DATA/) contains MRI scans acquired using seven different scanners, from two vendors, at three centers. Thus, the spatial resolution of its images vary across different patient exams. For instance, in the case of the axial T2W scans, the most common voxel spacing (in mm/voxel) observed is 3.0×0.5×0.5 (43%), followed by 3.6×0.3×0.3 (25%), 3.0×0.342×0.342 (15%) and others (17%). As a naive approach, we simply resample all scans to 3.0×0.5×0.5 mm/voxel.

- **Cropping to Region-of-Interest**: We naively assume that the prostate gland is typically located within the centre of every prostate MRI scan. Hence, we take a centre crop of each scan, measuring 20×256×256 voxels in dimensions. Note, this assumption does not hold true for the entirety of the [PI-CAI: Public Training and Development Dataset](https://pi-cai.grand-challenge.org/DATA/), where the prostate gland is off-center in several cases.

After following all the steps listed under sections ['Folder Structure'](README.md#folder-structure) and ['Data Preparation'](README.md#data-preparation), set your target paths in [`plan_overview.py`](src/picai_baseline/unet/plan_overview.py) and execute it:

```bash
python src/picai_baseline/unet/plan_overview.py
```

This command creates `.json`-based lists of every scan and its corresponding details (e.g. patient ID, study ID, paths to its imaging and annotation files) used in each split (training or validation split) per fold during 5-fold cross-validation, and stores them in `/workdir/results/UNet/overviews/`. These lists are subsequently used by the U-Net's data loaders during training. For example, lists used to complete the first fold of cross-validation would be: `PI-CAI_train-fold-0.json` and `PI-CAI_val-fold-0.json`.


### U-Net - Training and Cross-Validation
The overall framework for training this U-Net has been set up using various modular components from the [`monai`](https://github.com/Project-MONAI/MONAI) module (e.g. U-Net architecture, template for training) and the [`batchgenerators`](https://github.com/MIC-DKFZ/batchgenerators) module (e.g. data loaders, data augmentation policy as incorporated in the [nnU-Net](nnunet_baseline.md)). To train the model, run the following command:

```bash
python -u src/picai_baseline/unet/train.py \
  --weights_dir='/workdir/results/UNet/weights/' \
  --overviews_dir='/workdir/results/UNet/overviews/' \
  --folds 0 1 2 3 4 --max_threads 6 --enable_da 1 --num_epochs 250 \
  --validate_n_epochs 1 --validate_min_epoch 0
```
⚠️ If you are running this command inside a Docker container, please make sure that your container has at least 16 GB of shared memory space. Otherwise, you may run into issues with data generators and multithreading, as documented [here](https://grand-challenge.org/forums/forum/pi-cai-607/topic/single-and-multithreadedaugmenter-attribute-error-while-training-unet-1085/).

Full list of all available training arguments can be found in [`train.py`](src/picai_baseline/unet/train.py). Here is a summary of the arguments used in the command above:

| Argument                 | Meaning                |
|:-------------------------|:-----------------------|
| ```weights_dir```        | Directory to store model weights/checkpoints and an overview of performance metrics at train-time. |
| ```enable_da```        | Enable data augmentations (simplified policy adapted from the [nnU-Net](nnunet_baseline.md)).  |
| ```overviews_dir```  | Directory from which [overview lists](#u-net---data-preparation) are loaded (which define all images and annotations used per split per cross-validation fold). |
|```validate_min_epoch```          | Minimum number of epochs after which evaluation is performed using the validation split. Performance metrics and model weights are stored after this point. |
| ```max_threads```   | Number of CPU threads/cores to be used to parallelize data loaders. |
|```validate_n_epochs```   | Number of epochs that define the waiting period between two consecutive rounds of evaluation. |
| ```focal_loss_gamma```   | Value of the _gamma_ parameter in the [focal loss (FL)]((https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)) function used at train-time. When _gamma_ is set to 0, FL reduces down to weighted cross-entropy loss.  |
| ```folds```         | Cross-validation folds to be completed sequentially during training. E.g. `--folds 0` for a single fold, or `--folds 0 1 2 3 4` for all five folds |
|```num_epochs```           | Number of epochs that define the total training period. |

Additionally, note:

- In this baseline, the [segmentation output of the U-Net is processed via dynamic lesion extraction, to produce the csPCa detection map](https://github.com/DIAGNijmegen/picai_eval/#evaluate-softmax-volumes-instead-of-detection-maps). The [maximum value of this detection map is used as the case-level likelihood score for csPCa diagnosis](https://github.com/DIAGNijmegen/picai_eval/#evaluate-individual-detection-maps-with-python).

- Intensity normalization is performed at train-time, during data loading. For this, we apply instance-wise [_z_-score normalization](https://www.statology.org/z-score-normalization/) (where 0.5 and 99.5 percentiles of all intensity values are also used for clipping) to all three MRI sequences (T2W, ADC, DWI) independently. To add/use any other intensity normalization strategy, please make changes in [`training_setup/image_reader.py`](src/picai_baseline/unet/training_setup/image_reader.py) accordingly.

- Each epoch is defined as an iteration over 100 mini-batches of data (irregardless of how samples constitute one mini-batch). For reference, [nnU-Net](nnunet_baseline.md) defines each epoch as an iteration over 250 mini-batches of data.

- Evaluation of the validation split at train-time is currently performed using the ranking score of the [PI-CAI challenge](https://pi-cai.grand-challenge.org/) (via the [`picai_eval`](https://github.com/DIAGNijmegen/picai_eval/) module). To add/use any other performance metric(s), please make changes in [`training_setup/callbacks.py`](src/picai_baseline/unet/training_setup/callbacks.py) accordingly.

- Model training can be restarted/resumed using the `resume_training` argument. By default, this is set as `True`, and hence training will always resume if the same `weights_dir` argument is used, in combination with the same `folds` and `model_type` arguments (add `--resume_training 0` to disable).

- A simplified adaptation of the data augmentation policy from the [nnU-Net](nnunet_baseline.md) is used. To add/use your own transformations/policy for data augmentation, please make changes in [`training_setup/augmentations/nnUNet_DA.py`](src/picai_baseline/unet/training_setup/augmentations/nnUNet_DA.py) accordingly.

- Model weights are stored during training only when evaluation is performed and the current best validation metric has been surpassed (to save disk space we don't store every model). To store model weights through every training epoch or add your own calculation to determine which model is stored, please make changes in [`training_setup/callbacks.py`](src/picai_baseline/unet/training_setup/callbacks.py) accordingly.

- This U-Net model can be trained using the [Adam optimizer](https://arxiv.org/abs/1412.6980) and [focal loss (FL) function](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf). For FL, _gamma_ is set as 1.0 by default (note, when set as 0.0, FL reduces down to weighted cross-entropy loss). Meanwhile, _alpha_ is automatically set as the inverse of the case-level class balance for the training split per fold (voxel-level class balance is too severe to be used). For Adam, [_"amsgrad"_](https://openreview.net/forum?id=ryQu7f-RZ) functionality is set as True. To add/use any other optimizer(s) or loss function(s), please make 
changes in [`train.py`](src/picai_baseline/unet/train.py) and [`training_setup/loss_functions/`](src/picai_baseline/unet/training_setup/loss_functions) accordingly.

- This training framework supports the [U-Net architecture, as defined by the `monai` module](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unet.py). To add/use any other architecture(s), please make changes in [`train.py`](src/picai_baseline/unet/train.py) and [`training_setup/neural_networks/`](src/picai_baseline/unet/training_setup/neural_networks), [`training_setup/neural_network_selector.py`](src/picai_baseline/unet/training_setup/neural_network_selector.py) and [`training_setup/default_hyperparam.py`](src/picai_baseline/unet/training_setup/default_hyperparam.py) accordingly.

- The configuration of the U-Net architecture (i.e. downsampling strides and number of features per resolution) and the choice of the batch size is selected assuming only a single, typical desktop-class NVIDIA GPU (with 8-11 GB VRAM) is available. If more/less GPU VRAM is available, please feel free to make changes in [`training_setup/default_hyperparam.py`](src/picai_baseline/unet/training_setup/default_hyperparam.py) accordingly.


### U-Net - Inference Algorithm Submission to Grand Challenge
Once training is complete, there should be a single model checkpoint file (in `.pt` format) per fold, stored in the `weights_dir` that was specified at train-time. If the default command (noted in section [**'U-Net - Training and Cross-Validation'**](#u-net---training-and-cross-validation)) is used, then one of these should be `'/workdir/results/UNet/weights/unet_F0.pt'`. Given that this checkpoint not only includes the trained model weights, but also the optimizer state and epoch number (which are used to resume training), its memory footprint can be quite large. Before preparing our Docker container for the algorithm, we should trim down its size and store only what we need for deployment. Please apply the following function to every model checkpoint file that you plan to use in your grand-challenge.org algorithm:

```python
def process_model_weights(input_ckpt_path, output_ckpt_path):
    '''
    Loads model checkpoint that was stored at train-time, discards
    "optimizer_state_dict" and "epoch", and only keeps the trained 
    model weights (i.e. "model_state_dict"). Reduces memory footprint
    of weights file by nearly 5x.
    '''
    checkpoint = torch.load(input_ckpt_path, map_location=torch.device('cpu'))
    torch.save({
        'model_state_dict': checkpoint['model_state_dict']}, output_ckpt_path)
```

If you're using an ensemble, this function should be applied to each checkpoint file per member model (e.g. an ensemble can consist of the five models derived from all five folds of training/cross-validation). 

Next, we highly recommend completing the [full tutorial on how to create algorithms on grand-challenge.org](https://grand-challenge.org/documentation/create-your-own-algorithm/). In accordance with the same, we've built an [**example algorithm for you to use/adapt here**](https://github.com/DIAGNijmegen/picai_unet_gc_algorithm). Head over to the ["AI: Algorithm Submissions" page](https://pi-cai.grand-challenge.org/ai-algorithm-submissions/) on the challenge website for more details, and the final steps needed to make a submission to the PI-CAI challenge.
