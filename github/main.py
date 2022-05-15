import os
import random
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
import numpy as np

import hydra
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter

from datasets.cityscapes_Dataset import City_Dataset, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_Dataset
from datasets.synthia_Dataset import SYNTHIA_Dataset
from datasets.idd_dataset import IDDDataSet
from datasets.vistas_dataset import MapillaryDataSet
from perturbations.augmentations import augment, get_augmentation
from perturbations.fourier import fourier_mix
from perturbations.cutmix import cutmix_combine
from models import get_model
from models.ema import EMA
from utils.eval import Eval, synthia_set_16, synthia_set_13
import copy

class Trainer():
    def __init__(self, cfg, logger, writer):

        # Args
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.logger = logger
        self.writer = writer

        # Counters
        self.epoch = 0
        self.iter = 0
        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_source_MIou = 0

        # Metrics
        self.evaluator = Eval(self.cfg.data.num_classes)

        # Loss
        self.ignore_index = -1
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # Model
        self.model, params = get_model(self.cfg)
        # self.model = nn.DataParallel(self.model, device_ids=[0])  # TODO: test multi-gpu
        self.model.to(self.device)

        # # Previous model
        # self.ref_model0 = copy.deepcopy(self.model)  # reference model for knowledge distillation
        # saved_state_dict0 = torch.load("/data/shinahyung/code/3CUDA/0509/pixmatch_t1_tm/outputs/2022-04-27/03-55-57/best.pth", map_location=self.device)
        # self.ref_model0.load_state_dict(saved_state_dict0, strict=False)
        # # new_params0 = self.model.state_dict().copy()
        # # for i in saved_state_dict0:
        # #     if i in new_params0.keys():
        # #         new_params0[i] = saved_state_dict0[i]
        # # self.ref_model0.load_state_dict(new_params0)
        # for params0 in self.ref_model0.parameters():
        #     params0.requires_grad = False
        # self.ref_model0.eval()

        self.ref_model1 = copy.deepcopy(self.model)  # reference model for knowledge distillation
        checkpoint1 = torch.load("/data/shinahyung/code/3CUDA/0509/pixmatch_t1_tm/outputs/2022-04-27/03-55-57/best.pth", map_location=self.device)
        new_params1 = self.model.state_dict().copy()
        # if not self.cfg.train and 'shadow' in checkpoint1:
        #     state_dict1 = checkpoint1['shadow']
        # elif 'state_dict' in checkpoint1:
        state_dict1 = checkpoint1['state_dict']
        # else:
        #     state_dict1 = checkpoint1
        state_dict1 = {k.replace('module.', ''): v for k,
                      v in state_dict1.items()}
        for i in state_dict1:
            if i in new_params1.keys():
                new_params1[i] = state_dict1[i]
        self.ref_model1.load_state_dict(new_params1)
        for params1 in self.ref_model1.parameters():
            params1.requires_grad = False
        self.ref_model1.eval()

        # EMA
        self.ema = EMA(self.model, self.cfg.ema_decay)

        # Optimizer
        if self.cfg.opt.kind == "SGD":
            self.optimizer = torch.optim.SGD(
                params, momentum=self.cfg.opt.momentum, weight_decay=self.cfg.opt.weight_decay)
        elif self.cfg.opt.kind == "Adam":
            self.optimizer = torch.optim.Adam(params, betas=(
                0.9, 0.99), weight_decay=self.cfg.opt.weight_decay)
        else:
            raise NotImplementedError()
        self.lr_factor = 10

        
        # Source
        if self.cfg.data.source.dataset == 'synthia':
            source_train_dataset = SYNTHIA_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = SYNTHIA_Dataset(split='val', **self.cfg.data.source.kwargs)
        elif self.cfg.data.source.dataset == 'gta5':
            source_train_dataset = GTA5_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = GTA5_Dataset(split='val', **self.cfg.data.source.kwargs)
        else:
            raise NotImplementedError()
        self.source_dataloader = DataLoader(
            source_train_dataset, shuffle=True, drop_last=True, **self.cfg.data.loader.kwargs)
        self.source_val_dataloader = DataLoader(
            source_val_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)

        # Target
        if self.cfg.data.target.dataset == 'cityscapes':
            target_train_dataset = City_Dataset(split='train', **self.cfg.data.target.kwargs)
            target_val_dataset = City_Dataset(split='val', **self.cfg.data.target.kwargs)
        elif self.cfg.data.target.dataset == 'IDD':
            target_train_dataset = IDDDataSet(split='train', **self.cfg.data.target.kwargs)
            target_val_dataset = IDDDataSet(split='val', **self.cfg.data.target.kwargs)
        elif self.cfg.data.target.dataset == 'MapillaryVistas':
            target_train_dataset = MapillaryDataSet(split='train', **self.cfg.data.target.kwargs)
            target_val_dataset = MapillaryDataSet(split='val', **self.cfg.data.target.kwargs)
        else:
            raise NotImplementedError()
        self.target_dataloader = DataLoader(
            target_train_dataset, shuffle=True, drop_last=True, **self.cfg.data.loader.kwargs)
        self.target_val_dataloader = DataLoader(
            target_val_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)


        # Perturbations
        if self.cfg.lam_aug > 0:
            self.aug = get_augmentation()

    def train(self):

        # Loop over epochs
        self.continue_training = True
        while self.continue_training:

            # Train for a single epoch
            self.train_one_epoch()
            self.epoch += 1

        # Save final checkpoint
        self.logger.info("=> The best MIou was {:.3f} at iter {}".format(
            self.best_MIou, self.best_iter))
        self.logger.info(
            "=> Saving the final checkpoint to {}".format('final.pth'))
        self.save_checkpoint('final.pth')

    def train_one_epoch(self):


        # Helper
        def unpack(x):
            return (x[0], x[1]) if isinstance(x, tuple) else (x, None)

        # Training loop
        total = min(len(self.source_dataloader), len(self.target_dataloader))
        for batch_idx, (batch_s, batch_t) in enumerate(tqdm(
            zip(self.source_dataloader, self.target_dataloader),
            total=total, desc=f"Epoch {self.epoch + 1}"
        )):            
            # Load and reset
            self.model.train()
            self.evaluator.reset()

            # Learning rate
            self.poly_lr_scheduler(optimizer=self.optimizer)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]["lr"], self.iter)

            # Losses
            losses = {}

            ##########################
            # Source supervised loss #
            ##########################
            x, y, _ = batch_s

            if True:  # For VS Code collapsing

                # Data
                x = x.to(self.device)
                y = y.squeeze(dim=1).to(device=self.device,
                                        dtype=torch.long, non_blocking=True)

                # Fourier mix: source --> target
                if self.cfg.source_fourier:
                    x = fourier_mix(src_images=x, tgt_images=batch_t[0].to(
                        self.device), L=self.cfg.fourier_beta)

                # Forward
                pred = self.model(x, forward_target=self.cfg.num_target)
                pred_1, pred_2 = unpack(pred)

                # Loss (source)
                loss_source_1 = self.loss(pred_1, y)
                if self.cfg.aux:
                    loss_source_2 = self.loss(pred_2, y) * self.cfg.lam_aux
                    loss_source = loss_source_1 + loss_source_2
                else:
                    loss_source = loss_source_1

                # Backward
                loss_source.backward()

                # Clean up
                losses['source_main'] = loss_source_1.cpu().item()
                if self.cfg.aux:
                    losses['source_aux'] = loss_source_2.cpu().item()
                del x, y, loss_source, loss_source_1, loss_source_2


            ######################
            # Target Pseudolabel #
            ######################
            x, gt, id = batch_t
            x = x.to(self.device)

            ##############################
            # Pseudolabel - use all task #
            ##############################
            with torch.no_grad():
                ################################ Pseudo1
                pseudo_outputs0_1, pseudo_outputs0_2 = self.ref_model1(x.to(self.device), forward_target=0) # Source (S)
                pseudo_outputs1_1, pseudo_outputs1_2 = self.ref_model1(x.to(self.device), forward_target=1)
                pseudo_outputs2_1, pseudo_outputs2_2 = self.model(x.to(self.device), forward_target=self.cfg.num_target) # Source + TM3

                pseudo0_1 = F.softmax(pseudo_outputs0_1, dim=1)
                pseudo1_1 = F.softmax(pseudo_outputs1_1, dim=1)
                pseudo2_1 = F.softmax(pseudo_outputs2_1, dim=1)
                pseudo_best=torch.max(pseudo0_1,pseudo1_1)
                pseudo_best = torch.max(pseudo_best, pseudo2_1)

                label_best = torch.argmax(pseudo_best.detach(), dim=1)
                maxpred_best, argpred_best = torch.max(pseudo_best.detach(), dim=1)

                T = 0.5
                mask_1 = (maxpred_best > T)
                ignore_tensor = torch.ones(1).to(
                    self.device, dtype=torch.long) * self.ignore_index
                label_1 = torch.where(mask_1, label_best, ignore_tensor)
                     
                ################################ Pseudo2

                pseudo0_2 = F.softmax(pseudo_outputs0_2, dim=1)
                pseudo1_2 = F.softmax(pseudo_outputs1_2, dim=1)
                pseudo2_2 = F.softmax(pseudo_outputs2_2, dim=1)
                pseudo_best2=torch.max(pseudo0_2,pseudo1_2)
                pseudo_best2 = torch.max(pseudo_best2,pseudo2_2)
                
                label_best2 = torch.argmax(pseudo_best2.detach(), dim=1)
                maxpred_best2, argpred_best2 = torch.max(pseudo_best2.detach(), dim=1)

                T = 0.5
                mask_2 = (maxpred_best2 > T)
                ignore_tensor = torch.ones(1).to(
                    self.device, dtype=torch.long) * self.ignore_index
                label_2 = torch.where(mask_2, label_best2, ignore_tensor)

################################ plot
                # pseudo_ = label_1[0].unsqueeze(0)
                # print(pseudo_.shape)
                # # pseudo_ = torch.max(pseudo_, dim=1)[1]
                # pseudo_ = pseudo_.detach().cpu().numpy()
                # pseudo_ = np.asarray(pseudo_, dtype=np.uint8).squeeze(0)
                # output_col = colorize_mask(pseudo_)
                # output_col.save('/data/shinahyung/code/3CUDA/0509/target1/p2_%s_color.png' % (batch_idx))
################################


            pred = self.model(x.to(self.device), forward_target=self.cfg.num_target)
            pred_1_g, pred_2_g = unpack(pred)

            loss_self_train1 = self.loss(pred_1_g, label_1) * self.cfg.lam_new
            loss_self_train2 = self.cfg.lam_aux * self.loss(pred_2_g, label_2) * self.cfg.lam_new
            loss_self_train = loss_self_train1 + loss_self_train2

            loss_self_train.backward()
            del pred, pred_1_g, pred_2_g, loss_self_train1, loss_self_train2, loss_self_train

            # # # First step: run non-augmented image though model to get predictions
            # with torch.no_grad():

            #     # Substep 1: forward pass
            #     pred = self.model(x.to(self.device), forward_target=self.cfg.num_target)
            #     pred_1, pred_2 = unpack(pred)

            #     # Substep 2: convert soft predictions to hard predictions
            #     pred_P_1 = F.softmax(pred_1, dim=1)
            #     label_1 = torch.argmax(pred_P_1.detach(), dim=1)
            #     maxpred_1, argpred_1 = torch.max(pred_P_1.detach(), dim=1)
            #     T = self.cfg.pseudolabel_threshold
            #     mask_1 = (maxpred_1 > T)
            #     ignore_tensor = torch.ones(1).to(
            #         self.device, dtype=torch.long) * self.ignore_index
            #     label_1 = torch.where(mask_1, label_1, ignore_tensor)
            #     if self.cfg.aux:
            #         pred_P_2 = F.softmax(pred_2, dim=1)
            #         maxpred_2, argpred_2 = torch.max(pred_P_2.detach(), dim=1)
            #         pred_c = (pred_P_1 + pred_P_2) / 2
            #         maxpred_c, argpred_c = torch.max(pred_c, dim=1)
            #         mask = (maxpred_1 > T) | (maxpred_2 > T)
            #         label_2 = torch.where(mask, argpred_c, ignore_tensor)


            ############
            # Aug loss #
            ############
            if self.cfg.lam_aug > 0:

                # Second step: augment image and label
                x_aug, y_aug_1 = augment(
                    images=x.cpu(), labels=label_1.detach().cpu(), aug=self.aug)
                y_aug_1 = y_aug_1.to(device=self.device, non_blocking=True)
                if self.cfg.aux:
                    _, y_aug_2 = augment(
                        images=x.cpu(), labels=label_2.detach().cpu(), aug=self.aug)
                    y_aug_2 = y_aug_2.to(device=self.device, non_blocking=True)

                # Third step: run augmented image through model to get predictions
                pred_aug = self.model(x_aug.to(self.device), forward_target=self.cfg.num_target)
                pred_aug_1, pred_aug_2 = unpack(pred_aug)

                # Fourth step: calculate loss
                loss_aug_1 = self.loss(pred_aug_1, y_aug_1) * self.cfg.lam_aug
                if self.cfg.aux:
                    loss_aug_2 = self.loss(pred_aug_2, y_aug_2) * self.cfg.lam_aug * self.cfg.lam_aux
                    loss_aug = loss_aug_1 + loss_aug_2
                else:
                    loss_aug = loss_aug_1

                # Backward
                loss_aug.backward()

                # Clean up
                losses['aug_main'] = loss_aug_1.cpu().item()
                if self.cfg.aux:
                    losses['aug_aux'] = loss_aug_2.cpu().item()
                del pred_aug, pred_aug_1, pred_aug_2, loss_aug, loss_aug_1, loss_aug_2 #, loss_self_train, loss_self_train1, loss_self_train2

            # ################
            # # Fourier Loss #
            # ################
            # if self.cfg.lam_fourier > 0:

            #     # Second step: fourier mix
            #     x_fourier = fourier_mix(
            #         src_images=x.to(self.device),
            #         tgt_images=batch_s[0].to(self.device),
            #         L=self.cfg.fourier_beta)

            #     # Third step: run mixed image through model to get predictions
            #     pred_fourier = self.model(x_fourier.to(self.device))
            #     pred_fourier_1, pred_fourier_2 = unpack(pred_fourier)

            #     # Fourth step: calculate loss
            #     loss_fourier_1 = self.loss(pred_fourier_1, label_1) * \
            #         self.cfg.lam_fourier

            #     if self.cfg.aux:
            #         loss_fourier_2 = self.loss(pred_fourier_2, label_2) * \
            #             self.cfg.lam_fourier * self.cfg.lam_aux
            #         loss_fourier = loss_fourier_1 + loss_fourier_2
            #     else:
            #         loss_fourier = loss_fourier_1

            #     # Backward
            #     loss_fourier.backward()

            #     # Clean up
            #     losses['fourier_main'] = loss_fourier_1.cpu().item()
            #     if self.cfg.aux:
            #         losses['fourier_aux'] = loss_fourier_2.cpu().item()
            #     del pred_fourier, pred_fourier_1, pred_fourier_2, loss_fourier, loss_fourier_1, loss_fourier_2

            # ###############
            # # CutMix Loss #
            # ###############
            # if self.cfg.lam_cutmix > 0:

            #     # Second step: CutMix
            #     x_cutmix, y_cutmix = cutmix_combine(
            #         images_1=x,
            #         labels_1=label_1.unsqueeze(dim=1),
            #         images_2=batch_s[0].to(self.device),
            #         labels_2=batch_s[1].unsqueeze(dim=1).to(self.device, dtype=torch.long))
            #     y_cutmix = y_cutmix.squeeze(dim=1)

            #     # Third step: run mixed image through model to get predictions
            #     pred_cutmix = self.model(x_cutmix)
            #     pred_cutmix_1, pred_cutmix_2 = unpack(pred_cutmix)

            #     # Fourth step: calculate loss
            #     loss_cutmix_1 = self.loss(pred_cutmix_1, y_cutmix) * \
            #         self.cfg.lam_cutmix
            #     if self.cfg.aux:
            #         loss_cutmix_2 = self.loss(pred_cutmix_2, y_cutmix) * \
            #             self.cfg.lam_cutmix * self.cfg.lam_aux
            #         loss_cutmix = loss_cutmix_1 + loss_cutmix_2
            #     else:
            #         loss_cutmix = loss_cutmix_1

            #     # Backward
            #     loss_cutmix.backward()

            #     # Clean up
            #     losses['cutmix_main'] = loss_cutmix_1.cpu().item()
            #     if self.cfg.aux:
            #         losses['cutmix_aux'] = loss_cutmix_2.cpu().item()
            #     del pred_cutmix, pred_cutmix_1, pred_cutmix_2, loss_cutmix, loss_cutmix_1, loss_cutmix_2


            ###############
            # Validation  #
            ###############
            if self.epoch>=0 and batch_idx>0 and batch_idx % 500 == 0:

                # Use EMA params to evaluate performance
                self.ema.apply_shadow()
                self.ema.model.eval()
                self.ema.model.cuda()

                # Validate on source (if possible) and target
                # if self.cfg.data.source_val_iterations > 0:
                #     self.validate(mode='source')
                PA, MPA, MIoU, FWIoU = self.validate()

                # Restore current (non-EMA) params for training
                self.ema.restore()

                # Log val results
                self.writer.add_scalar('PA', PA, self.epoch)
                self.writer.add_scalar('MPA', MPA, self.epoch)
                self.writer.add_scalar('MIoU', MIoU, self.epoch)
                self.writer.add_scalar('FWIoU', FWIoU, self.epoch)

                # Save checkpoint if new best model
                self.current_MIoU = MIoU
                is_best = MIoU > self.best_MIou
                if is_best:
                    self.best_MIou = MIoU
                    self.best_iter = self.iter
                    self.logger.info("=> Saving a new best checkpoint...")
                    self.logger.info("=> The best val MIoU is now {:.3f} from iter {}".format(
                        self.best_MIou, self.best_iter))
                    self.save_checkpoint('best.pth')
                else:
                    self.logger.info("=> The MIoU of val did not improve.")
                    self.logger.info("=> The best val MIoU is still {:.3f} from iter {}".format(
                        self.best_MIou, self.best_iter))


            # Step optimizer if accumulated enough gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update model EMA parameters each step
            self.ema.update_params()

            # Calculate total loss
            total_loss = sum(losses.values())

            # Log main losses
            for name, loss in losses.items():
                self.writer.add_scalar(f'train/{name}', loss, self.iter)

            # Log
            if batch_idx % 100 == 0:
                log_string = f"[Epoch {self.epoch}]\t"
                log_string += '\t'.join([f'{n}: {l:.3f}' for n, l in losses.items()])
                self.logger.info(log_string)

            # Increment global iteration counter
            self.iter += 1

            # End training after finishing iterations
            if self.iter > self.cfg.opt.iterations:
                self.continue_training = False
                return


        # After each epoch, update model EMA buffers (i.e. batch norm stats)
        self.ema.update_buffer()

    @ torch.no_grad()
    def validate(self, mode='target'):
        """Validate on target"""
        self.logger.info('Validating')
        self.evaluator.reset()
        self.model.eval()

        # Select dataloader
        if mode == 'target':
            val_loader = self.target_val_dataloader
        elif mode == 'source':
            val_loader = self.source_val_dataloader
        else:
            raise NotImplementedError()

        # Loop
        for val_idx, (x, y, id) in enumerate(tqdm(val_loader, desc=f"Val Epoch {self.epoch + 1}")):
            if mode == 'source' and val_idx >= self.cfg.data.source_val_iterations:
                break

            # Forward
            x = x.to(self.device)
            y = y.to(device=self.device, dtype=torch.long)
            pred = self.model(x, forward_target=self.cfg.num_target)
            if isinstance(pred, tuple):
                pred = pred[0]

            # Convert to numpy
            label = y.squeeze(dim=1).cpu().numpy()
            argpred = np.argmax(pred.data.cpu().numpy(), axis=1)

            # Add to evaluator
            self.evaluator.add_batch(label, argpred)

        # Tensorboard images
        vis_imgs = 2
        images_inv = inv_preprocess(x.clone().cpu(), vis_imgs, numpy_transform=True)
        labels_colors = decode_labels(label, vis_imgs)
        preds_colors = decode_labels(argpred, vis_imgs)
        for index, (img, lab, predc) in enumerate(zip(images_inv, labels_colors, preds_colors)):
            self.writer.add_image(str(index) + '/images', img, self.epoch)
            self.writer.add_image(str(index) + '/labels', lab, self.epoch)
            self.writer.add_image(str(index) + '/preds', predc, self.epoch)

        # Calculate and log
        if self.cfg.data.source.kwargs.class_16:
            PA = self.evaluator.Pixel_Accuracy()
            MPA_16, MPA_13 = self.evaluator.Mean_Pixel_Accuracy()
            MIoU_16, MIoU_13 = self.evaluator.Mean_Intersection_over_Union()
            FWIoU_16, FWIoU_13 = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            PC_16, PC_13 = self.evaluator.Mean_Precision()
            self.logger.info('Epoch:{:.3f}, PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(
                self.epoch, PA, MPA_16, MIoU_16, FWIoU_16, PC_16))
            self.logger.info('Epoch:{:.3f}, PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(
                self.epoch, PA, MPA_13, MIoU_13, FWIoU_13, PC_13))
            self.writer.add_scalar('PA', PA, self.epoch)
            self.writer.add_scalar('MPA_16', MPA_16, self.epoch)
            self.writer.add_scalar('MIoU_16', MIoU_16, self.epoch)
            self.writer.add_scalar('FWIoU_16', FWIoU_16, self.epoch)
            self.writer.add_scalar('MPA_13', MPA_13, self.epoch)
            self.writer.add_scalar('MIoU_13', MIoU_13, self.epoch)
            self.writer.add_scalar('FWIoU_13', FWIoU_13, self.epoch)
            PA, MPA, MIoU, FWIoU = PA, MPA_13, MIoU_13, FWIoU_13
        else:
            PA = self.evaluator.Pixel_Accuracy()
            MPA = self.evaluator.Mean_Pixel_Accuracy()
            MIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            PC = self.evaluator.Mean_Precision()
            self.logger.info('Epoch:{:.3f}, PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(
                self.epoch, PA, MPA, MIoU, FWIoU, PC))
            self.writer.add_scalar('PA', PA, self.epoch)
            self.writer.add_scalar('MPA', MPA, self.epoch)
            self.writer.add_scalar('MIoU', MIoU, self.epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.epoch)
        return PA, MPA, MIoU, FWIoU



    @ torch.no_grad()
    def store_old_prediction(self, mode='target'):
        """Validate on target"""
        self.logger.info('Validating')
        self.evaluator.reset()
        self.model.eval()

        # Select dataloader
        if mode == 'target':
            val_loader = self.target_dataloader

        old_lst = {}
        # Loop
        for val_idx, (x, y, id) in enumerate(tqdm(val_loader, desc=f"Val Epoch {self.epoch + 1}")):
            if mode == 'source' and val_idx >= self.cfg.data.source_val_iterations:
                break

            # Forward
            if val_idx % 100 ==0:
                print(val_idx)
            x = x.to(self.device)
            y = y.to(device=self.device, dtype=torch.long)
            pred = self.model(x, forward_target=self.cfg.num_target)
            if isinstance(pred, tuple):
                pred = pred[0]

            pred_s = pred.detach().cpu().numpy()
            old_lst[id[0]] = pred_s
        # np.save('/data/shinahyung/code/3CUDA/0509/target1_new/t1_old.npy', old_lst)
        print("Load DONE!!!!!!!!!!!!!!!!!")
        self.old_lst = old_lst
        
        del old_lst
        torch.cuda.empty_cache()


    def save_checkpoint(self, filename='checkpoint.pth'):
        torch.save({
            'epoch': self.epoch + 1,
            'iter': self.iter,
            'state_dict': self.ema.model.state_dict(),
            'shadow': self.ema.shadow,
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.best_MIou
        }, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')

        # Get model state dict
        if not self.cfg.train and 'shadow' in checkpoint:
            state_dict = checkpoint['shadow']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove DP/DDP if it exists
        state_dict = {k.replace('module.', ''): v for k,
                      v in state_dict.items()}

        # Load state dict
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
        self.logger.info(f"Model loaded successfully from {filename}")

        # Load optimizer and epoch
        if self.cfg.train and self.cfg.model.resume_from_checkpoint:
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info(f"Optimizer loaded successfully from {filename}")
            if 'epoch' in checkpoint and 'iter' in checkpoint:
                self.epoch = checkpoint['epoch']
                self.iter = checkpoint['iter'] if 'iter' in checkpoint else checkpoint['iteration']
                self.logger.info(f"Resuming training from epoch {self.epoch} iter {self.iter}")
        else:
            self.logger.info(f"Did not resume optimizer")

    def poly_lr_scheduler(self, optimizer, init_lr=None, iter=None, max_iter=None, power=None):
        init_lr = self.cfg.opt.lr if init_lr is None else init_lr
        iter = self.iter if iter is None else iter
        max_iter = self.cfg.opt.iterations if max_iter is None else max_iter
        power = self.cfg.opt.poly_power if power is None else power
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        optimizer.param_groups[0]["lr"] = new_lr
        if len(optimizer.param_groups) == 2:
            optimizer.param_groups[1]["lr"] = 10 * new_lr


@hydra.main(config_path='configs', config_name='gta5')
def main(cfg: DictConfig):

    # Seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Monitoring
    if cfg.wandb:
        import wandb
        wandb.init(project='pixmatch', name=cfg.name, config=cfg, sync_tensorboard=True)
    writer = SummaryWriter(cfg.name)

    # Trainer
    trainer = Trainer(cfg=cfg, logger=logger, writer=writer)

    # Load pretrained checkpoint
    if cfg.model.checkpoint:
        assert Path(cfg.model.checkpoint).is_file(), f'not a file: {cfg.model.checkpoint}'
        trainer.load_checkpoint(cfg.model.checkpoint)

    # Print configuration
    logger.info('\n' + OmegaConf.to_yaml(cfg))

    # Train
    if cfg.train:
        trainer.train()
    # Evaluate
    else:
        # trainer.store_old_prediction()
        trainer.validate()
        trainer.evaluator.Print_Every_class_Eval(
            out_16_13=(int(cfg.data.num_classes) in [16, 13]))


if __name__ == '__main__':
    main()