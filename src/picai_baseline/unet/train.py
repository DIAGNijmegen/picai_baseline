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

import argparse
import ast

import torch
from picai_baseline.unet.training_setup.augmentations.nnUNet_DA import \
    apply_augmentations
from picai_baseline.unet.training_setup.callbacks import (
    optimize_model, resume_or_restart_training, validate_model)
from picai_baseline.unet.training_setup.compute_spec import \
    compute_spec_for_run
from picai_baseline.unet.training_setup.data_generator import prepare_datagens
from picai_baseline.unet.training_setup.default_hyperparam import \
    get_default_hyperparams
from picai_baseline.unet.training_setup.loss_functions.focal import FocalLoss
from picai_baseline.unet.training_setup.neural_network_selector import \
    neural_network_for_run
from torch.utils.tensorboard import SummaryWriter


def main():
    # command line arguments for hyperparameters and I/O paths
    parser = argparse.ArgumentParser(description='Command Line Arguments for Training Script')

    # data I/0 + experimental setup
    parser.add_argument('--max_threads', type=int, default=12,
                        help="Max threads/workers for data loaders")
    parser.add_argument('--validate_n_epochs', type=int, default=10,               
                        help="Trigger validation every N epochs")
    parser.add_argument('--validate_min_epoch', type=int, default=50,               
                        help="Trigger validation after minimum N epochs")
    parser.add_argument('--export_best_model', type=int, default=1,                
                        help="Export model checkpoints")
    parser.add_argument('--resume_training', type=str, default=1,                
                        help="Resume training model, if checkpoint exists")
    parser.add_argument('--weights_dir', type=str, required=True,            
                        help="Path to export model checkpoints")
    parser.add_argument('--overviews_dir', type=str, required=True,            
                        help="Base path to training/validation data sheets")
    parser.add_argument('--folds', type=int, nargs='+', required=True, 
                        help="Folds selected for training/validation run")

    # training hyperparameters
    parser.add_argument('--image_shape', type=int, nargs="+", default=[20, 256, 256],   
                        help="Input image shape (z, y, x)")
    parser.add_argument('--num_channels', type=int, default=3,                
                        help="Number of input channels/sequences")
    parser.add_argument('--num_classes', type=int, default=2,                
                        help="Number of classes at train-time")
    parser.add_argument('--num_epochs', type=int, default=100,              
                        help="Number of training epochs")
    parser.add_argument('--base_lr', type=float, default=0.001,            
                        help="Learning rate")
    parser.add_argument('--focal_loss_gamma', type=float, default=1.0,              
                        help="Focal Loss gamma value")
    parser.add_argument('--enable_da', type=int, default=1,                
                        help="Enable data augmentation")

    # neural network-specific hyperparameters
    parser.add_argument('--model_type', type=str, default='unet',                                                    
                        help="Neural network: architectures")
    parser.add_argument('--model_strides', type=str, default='[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]', 
                        help="Neural network: convolutional strides (as string representation)")
    parser.add_argument('--model_features', type=str, default='[32, 64, 128, 256, 512, 1024]',                           
                        help="Neural network: number of encoder channels (as string representation)")
    parser.add_argument('--batch_size', type=int, default=8,                                                         
                        help="Mini-batch size")
    parser.add_argument('--use_def_model_hp', type=int, default=1,                                                         
                        help="Use default set of model-specific hyperparameters")

    args = parser.parse_args()
    args.model_strides = ast.literal_eval(args.model_strides)
    args.model_features = ast.literal_eval(args.model_features)

    # retrieve default set of hyperparam (architecture, batch size) for given neural network
    if bool(args.use_def_model_hp):
        args = get_default_hyperparams(args)

    # for each fold
    for f in args.folds:
        # --------------------------------------------------------------------------------------------------------------------------
        # GPU/CPU specifications
        device, args = compute_spec_for_run(args=args)

        # derive dataLoaders
        train_gen, valid_gen, class_weights = prepare_datagens(args=args, fold_id=f)

        # integrate data augmentation pipeline from nnU-Net
        train_gen = apply_augmentations(
            dataloader=train_gen,
            num_threads=args.num_threads,
            disable=(not bool(args.enable_da))
        )
        
        # initialize multi-threaded augmenter in background
        train_gen.restart()

        # model definition
        model = neural_network_for_run(args=args, device=device)

        # loss function + optimizer
        loss_func = FocalLoss(alpha=class_weights[-1], gamma=args.focal_loss_gamma).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.base_lr, amsgrad=True)
        # --------------------------------------------------------------------------------------------------------------------------
        # training loop
        writer = SummaryWriter()

        # resume or restart training model, based on whether checkpoint exists
        model, optimizer, tracking_metrics = resume_or_restart_training(
            model=model, optimizer=optimizer,
            device=device, args=args, fold_id=f
        )

        # for each epoch
        for epoch in range(tracking_metrics['start_epoch'], args.num_epochs):

            # optimize model x N training steps + update learning rate
            model.train()
            tracking_metrics['epoch'] = epoch

            model, optimizer, train_gen, tracking_metrics, writer = optimize_model(
                model=model, optimizer=optimizer, loss_func=loss_func, train_gen=train_gen,
                args=args, tracking_metrics=tracking_metrics, device=device, writer=writer
            )

            # ----------------------------------------------------------------------------------------------------------------------
            # for each round of validation
            if ((epoch+1) % args.validate_n_epochs == 0) and ((epoch+1) >= args.validate_min_epoch):

                # validate model per N epochs + export model weights
                model.eval()
                with torch.no_grad():  # no gradient updates during validation

                    model, optimizer, valid_gen, tracking_metrics, writer = validate_model(
                        model=model, optimizer=optimizer, valid_gen=valid_gen, args=args,
                        tracking_metrics=tracking_metrics, device=device, writer=writer
                    )

        # --------------------------------------------------------------------------------------------------------------------------
        print(
            f"Training Complete! Peak Validation Ranking Score: {tracking_metrics['best_metric']:.4f} "
            f"@ Epoch: {tracking_metrics['best_metric_epoch']}")
        writer.close()
        # --------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
