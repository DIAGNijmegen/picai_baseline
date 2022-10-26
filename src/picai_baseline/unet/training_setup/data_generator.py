#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from collections import OrderedDict
from pathlib import Path

import monai
import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from monai.transforms import Compose, EnsureType
from picai_baseline.unet.training_setup.image_reader import SimpleITKDataset


def default_collate(batch):
    """collate multiple samples into batches, if needed"""

    if isinstance(batch[0], np.ndarray):
        return np.vstack(batch)
    elif isinstance(batch[0], (int, np.int64)):
        return np.array(batch).astype(np.int32)
    elif isinstance(batch[0], (float, np.float32)):
        return np.array(batch).astype(np.float32)
    elif isinstance(batch[0], (np.float64,)):
        return np.array(batch).astype(np.float64)
    elif isinstance(batch[0], (dict, OrderedDict)):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], torch.Tensor):
        return torch.vstack(batch)
    else:
        raise TypeError('unknown type for batch:', type(batch))


class DataLoaderFromDataset(DataLoader):
    """Create dataloader from given dataset"""

    def __init__(self, data, batch_size, num_threads, seed_for_shuffle=1, collate_fn=default_collate,
                 return_incomplete=False, shuffle=True, infinite=False):
        super(DataLoaderFromDataset, self).__init__(data, batch_size, num_threads, seed_for_shuffle,
                                                    return_incomplete=return_incomplete, shuffle=shuffle,
                                                    infinite=infinite)
        self.collate_fn = collate_fn
        self.indices = np.arange(len(data))

    def generate_train_batch(self):

        # randomly select N samples (N = batch size)
        indices = self.get_indices()

        # create dictionary per sample
        batch = [{'data': self._data[i][0].numpy(),
                  'seg': self._data[i][1].numpy()} for i in indices]

        return self.collate_fn(batch)


def prepare_datagens(args, fold_id):
    """Load data sheets --> Create datasets --> Create data loaders"""

    # load datasheets
    with open(Path(args.overviews_dir) / f'PI-CAI_train-fold-{fold_id}.json') as fp:
        train_json = json.load(fp)
    with open(Path(args.overviews_dir) / f'PI-CAI_val-fold-{fold_id}.json') as fp:
        valid_json = json.load(fp)

    # load paths to images and labels
    train_data = [np.array(train_json['image_paths']), np.array(train_json['label_paths'])]
    valid_data = [np.array(valid_json['image_paths']), np.array(valid_json['label_paths'])]

    # use case-level class balance to deduce required train-time class weights
    class_ratio_t = [int(np.sum(train_json['case_label'])), int(len(train_data[0])-np.sum(train_json['case_label']))]
    class_ratio_v = [int(np.sum(valid_json['case_label'])), int(len(valid_data[0])-np.sum(valid_json['case_label']))]
    class_weights = (class_ratio_t / np.sum(class_ratio_t))

    # log dataset definition
    print('Dataset Definition:', "-"*80)
    print(f'Fold Number: {fold_id}')
    print('Data Classes:', list(np.unique(train_json['case_label'])))
    print(f'Train-Time Class Weights: {class_weights}')
    print(f'Training Samples [-:{class_ratio_t[1]};+:{class_ratio_t[0]}]: {len(train_data[1])}')
    print(f'Validation Samples [-:{class_ratio_v[1]};+:{class_ratio_v[0]}]: {len(valid_data[1])}')

    # dummy dataloader for sanity check
    pretx = [EnsureType()]
    check_ds = SimpleITKDataset(image_files=train_data[0][:args.batch_size*2],
                                seg_files=train_data[1][:args.batch_size*2],
                                transform=Compose(pretx),
                                seg_transform=Compose(pretx))
    check_loader = DataLoaderFromDataset(check_ds, batch_size=args.batch_size, num_threads=args.num_threads)
    data_pair = monai.utils.misc.first(check_loader)
    print('DataLoader - Image Shape: ', data_pair['data'].shape)
    print('DataLoader - Label Shape: ', data_pair['seg'].shape)
    print("-"*100)

    assert args.image_shape == list(data_pair['data'].shape[2:])
    assert args.num_channels == data_pair['data'].shape[1]
    assert args.num_classes == len(np.unique(train_json['case_label']))

    # actual dataloaders used at train-time
    train_ds = SimpleITKDataset(image_files=train_data[0], seg_files=train_data[1],
                                transform=Compose(pretx),  seg_transform=Compose(pretx))
    valid_ds = SimpleITKDataset(image_files=valid_data[0], seg_files=valid_data[1],
                                transform=Compose(pretx),  seg_transform=Compose(pretx))
    train_ldr = DataLoaderFromDataset(train_ds, 
        batch_size=args.batch_size, num_threads=args.num_threads, infinite=True, shuffle=True)
    valid_ldr = DataLoaderFromDataset(valid_ds, 
        batch_size=args.batch_size, num_threads=1, infinite=False, shuffle=False)

    return train_ldr, valid_ldr, class_weights.astype(np.float32)
