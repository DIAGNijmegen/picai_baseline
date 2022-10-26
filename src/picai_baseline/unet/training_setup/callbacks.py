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

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from picai_baseline.unet.training_setup.poly_lr import poly_lr
from picai_eval import evaluate
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter


def resume_or_restart_training(model, optimizer, device, args, fold_id):
    """Resume/restart training, based on whether checkpoint exists"""

    weights_file = Path(args.weights_dir) / f"{args.model_type}_F{fold_id}.pt"
    metrics_file = Path(args.weights_dir) / f"{args.model_type}_F{fold_id}_metrics.xlsx"

    if bool(args.resume_training) and weights_file.is_file():
        print("Loading Weights From:", weights_file)
        checkpoint = torch.load(weights_file)

        # load weights and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.to(device)

        # load train-time metrics from interrupted run
        tracking_metrics = {}
        if metrics_file.is_file():

            saved_metrics = pd.read_excel(metrics_file, engine='openpyxl')
            all_epochs = (saved_metrics['epoch'].values).tolist()
            all_valid_metrics_auroc = (saved_metrics['valid_auroc'].values).tolist()
            all_valid_metrics_ap = (saved_metrics['valid_ap'].values).tolist()
            all_valid_metrics_ranking = (saved_metrics['valid_ranking'].values).tolist()

            tracking_metrics = {
                'fold_id':                   fold_id,
                'start_epoch':               checkpoint['epoch'] + 1,  # resume at next epoch
                'all_epochs':                all_epochs,
                'all_train_loss':           (saved_metrics['train_loss'].values).tolist(),
                'all_valid_metrics_auroc':   all_valid_metrics_auroc,
                'all_valid_metrics_ap':      all_valid_metrics_ap,
                'all_valid_metrics_ranking': all_valid_metrics_ranking,
                'best_metric':               np.max(all_valid_metrics_ranking),
                'best_metric_epoch':         all_epochs[all_valid_metrics_ranking.index(
                    np.max(all_valid_metrics_ranking))]}

            print('Previous Record of Metrics Loaded:', metrics_file)
        else:
            print('Previous Record of Metrics Not Found:', metrics_file)

            tracking_metrics = {
                'fold_id':                    fold_id,
                'start_epoch':                checkpoint['epoch'],
                'all_epochs':                 [],
                'all_train_loss':             [],
                'all_valid_metrics_auroc':    [],
                'all_valid_metrics_ap':       [],
                'all_valid_metrics_ranking':  [],
                'best_metric': -1,
                'best_metric_epoch': -1}

        print("Resume Training: Epoch",  tracking_metrics['start_epoch']+1)
        print("Best Validation Metric:", tracking_metrics['best_metric'],
              "@ Epoch", tracking_metrics['best_metric_epoch'])
    else:
        tracking_metrics = {
            'fold_id':                    fold_id,
            'start_epoch':                0,
            'all_epochs':                 [],
            'all_train_loss':             [],
            'all_valid_metrics_auroc':    [],
            'all_valid_metrics_ap':       [],
            'all_valid_metrics_ranking':  [],
            'best_metric': -1,
            'best_metric_epoch': -1}

        print("Start Training: Epoch", tracking_metrics['start_epoch']+1)

    return model, optimizer, tracking_metrics


def optimize_model(model, optimizer, loss_func, train_gen, args, tracking_metrics, device, writer):
    """Optimize model x N training steps per epoch + update learning rate"""

    train_loss, step = 0,  0
    start_time = time.time()
    epoch = tracking_metrics['epoch']

    # for each mini-batch or optimization step
    for batch_data in train_gen:
        step += 1
        try:
            inputs = batch_data['data'].to(device)
            labels = batch_data['seg'].to(device)
        except Exception:
            inputs = torch.from_numpy(batch_data['data']).to(device)
            labels = torch.from_numpy(batch_data['seg']).to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, labels[:, 0, ...].long())
        train_loss += loss.item()

        # backpropagate + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # define each training epoch == 100 steps (note: nnU-Net uses 250 steps)
        if step >= 100: 
            break

    # update learning rate
    updated_lr = poly_lr(epoch+1, args.num_epochs, args.base_lr, 0.95)
    optimizer.param_groups[0]['lr'] = updated_lr
    print("Learning Rate Updated! New Value: "+str(np.round(updated_lr, 10)), flush=True)

    # track training metrics
    train_loss /= step
    tracking_metrics['train_loss'] = train_loss
    writer.add_scalar("train_loss", train_loss, epoch+1)
    print("-" * 100)
    print(f"Epoch {epoch + 1}/{args.num_epochs} (Train. Loss: {train_loss:.4f}; \
        Time: {int(time.time()-start_time)}sec; Steps Completed: {step})", flush=True)

    return model, optimizer, train_gen, tracking_metrics, writer


def validate_model(model, optimizer, valid_gen, args, tracking_metrics, device, writer):
    """Validate model per N epoch + export model weights"""

    all_valid_preds, all_valid_labels = [], []
    epoch, f = tracking_metrics['epoch'], tracking_metrics['fold_id']

    # for each validation sample
    for valid_data in valid_gen:

        try:
            valid_images = valid_data['data'].to(device)
            valid_labels = valid_data['seg']
        except Exception:
            valid_images = torch.from_numpy(valid_data['data']).to(device)
            valid_labels = torch.from_numpy(valid_data['seg'])

        # test-time augmentation
        valid_images = [valid_images, torch.flip(valid_images, [4]).to(device)]

        # aggregate all validation predictions
        # gaussian blur to counteract checkerboard artifacts in
        # predictions from the use of transposed conv. in the U-Net
        preds = [
            torch.sigmoid(model(x))[:, 1, ...].detach().cpu().numpy()
            for x in valid_images
        ]

        # revert horizontally flipped tta image
        preds[1] = np.flip(preds[1], [3])

        # gaussian blur to counteract checkerboard artifacts in
        # predictions from the use of transposed conv. in the U-Net
        all_valid_preds += [
            np.mean([
                gaussian_filter(x, sigma=1.5)
                for x in preds
            ], axis=0)
        ]
        all_valid_labels += [valid_labels.numpy()[:, 0, ...]]

    # track validation metrics
    valid_metrics = evaluate(y_det=iter(np.concatenate([x for x in np.array(all_valid_preds)], axis=0)),
                             y_true=iter(np.concatenate([x for x in np.array(all_valid_labels)], axis=0)),
                             y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])

    num_pos = int(np.sum([np.max(y) for y in np.concatenate(
        [x for x in np.array(all_valid_labels)], axis=0)]))
    num_neg = int(len(np.concatenate([x for x in
                                      np.array(all_valid_labels)], axis=0)) - num_pos)

    tracking_metrics['all_epochs'].append(epoch+1)
    tracking_metrics['all_train_loss'].append(tracking_metrics['train_loss'])
    tracking_metrics['all_valid_metrics_auroc'].append(valid_metrics.auroc)
    tracking_metrics['all_valid_metrics_ap'].append(valid_metrics.AP)
    tracking_metrics['all_valid_metrics_ranking'].append(valid_metrics.score)

    # export train-time + validation metrics as .xlsx sheet
    metricsData = pd.DataFrame(list(zip(tracking_metrics['all_epochs'],
                                        tracking_metrics['all_train_loss'],
                                        tracking_metrics['all_valid_metrics_auroc'],
                                        tracking_metrics['all_valid_metrics_ap'],
                                        tracking_metrics['all_valid_metrics_ranking'])),
                               columns=['epoch', 'train_loss', 'valid_auroc', 'valid_ap', 'valid_ranking'])

    # create target folder and save exports sheet
    os.makedirs(args.weights_dir, exist_ok=True)

    metrics_file = Path(args.weights_dir) / f"{args.model_type}_F{f}_metrics.xlsx"
    metricsData.to_excel(metrics_file, encoding='utf-8', index=False)

    writer.add_scalar("valid_auroc",   valid_metrics.auroc, epoch+1)
    writer.add_scalar("valid_ap",      valid_metrics.AP,    epoch+1)
    writer.add_scalar("valid_ranking", valid_metrics.score, epoch+1)

    print(f"Valid. Performance [Benign or Indolent PCa (n={num_neg}) \
        vs. csPCa (n={num_pos})]:\nRanking Score = {valid_metrics.score:.3f},\
        AP = {valid_metrics.AP:.3f}, AUROC = {valid_metrics.auroc:.3f}", flush=True)

    # store model checkpoint if validation metric improves
    if valid_metrics.score > tracking_metrics['best_metric']:
        tracking_metrics['best_metric'] = valid_metrics.score
        tracking_metrics['best_metric_epoch'] = epoch + 1
        if bool(args.export_best_model):
            weights_file = Path(args.weights_dir) / f"{args.model_type}_F{f}.pt"

            print("Validation Ranking Score Improved! Saving New Best Model", flush=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, weights_file)
    return model, optimizer, valid_gen, tracking_metrics, writer
