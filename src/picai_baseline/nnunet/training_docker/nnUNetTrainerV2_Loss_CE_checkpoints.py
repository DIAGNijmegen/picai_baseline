#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from nnunet.training.network_training.nnUNet_variants.loss_function.nnUNetTrainerV2_Loss_CE import nnUNetTrainerV2_Loss_CE


class nnUNetTrainerV2_Loss_CE_checkpoints(nnUNetTrainerV2_Loss_CE):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.save_latest_only = False


class nnUNetTrainerV2_Loss_CE_checkpoints2(nnUNetTrainerV2_Loss_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        pass


class nnUNetTrainerV2_Loss_CE_checkpoints3(nnUNetTrainerV2_Loss_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        pass
