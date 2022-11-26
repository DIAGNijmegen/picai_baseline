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

from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.network_training.nnUNet_variants.loss_function.nnUNetTrainerV2_focalLoss import \
    FocalLoss
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from torch import nn

# TODO: replace FocalLoss by fixed implemetation (and set smooth=0 in that one?)


class FL_and_CE_loss(nn.Module):
    def __init__(self, fl_kwargs=None, ce_kwargs=None, alpha=0.5, aggregate="sum"):
        super(FL_and_CE_loss, self).__init__()
        if fl_kwargs is None:
            fl_kwargs = {}
        if ce_kwargs is None:
            ce_kwargs = {}

        self.aggregate = aggregate
        self.fl = FocalLoss(apply_nonlin=nn.Softmax(), **fl_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.alpha = alpha

    def forward(self, net_output, target):
        fl_loss = self.fl(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = self.alpha*fl_loss + (1-self.alpha)*ce_loss
        else:
            raise NotImplementedError("nah son")
        return result


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints(nnUNetTrainerV2):
    """
    Set loss to FL + CE and set checkpoints
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = FL_and_CE_loss(alpha=0.5)
        self.save_latest_only = False


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints2(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints3(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints4(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints5(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints6(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints7(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints8(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints9(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints10(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
