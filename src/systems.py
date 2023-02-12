import os
import copy
import math
import random
import dotmap
import numpy as np
from typing import Any
from textwrap import wrap
from dotmap import DotMap
from itertools import chain
from collections import OrderedDict
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
import wandb

from src.datasets import datasets
from src.models import resnet_small, resnet
from src.models.transfer import LogisticRegression
from src.models.viewmaker import Viewmaker
from src.objectives.adversarial import AdversarialSimCLRLoss
from src.objectives.divmaker import DivMakerLoss
from src.objectives.simclr import SimCLRObjective
from src.objectives.memory_bank import MemoryBank
from src.utils import utils
from src.datasets.data_statistics import get_data_mean_and_stdev


class PretrainSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.loss_name = self.config.loss_params.objective

        if self.config.loss_params.objective == "AdversarialSimCLRLoss":
            print("Using AdversarialSimCLRLoss")

        # Optionally optimize temperature.
        if self.config.loss_params.optim_t:
            self.t = torch.nn.Parameter(torch.tensor(self.config.loss_params.t), requires_grad=True)
        else:
            self.t = self.config.loss_params.t

        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
            config.data_params.default_augmentations or 'none',
            resize_imagenet_to_32=self.config.data_params.resize_imagenet_to_32 or False,
            mask=self.config.data_params.mask or False,
            zscore=self.config.data_params.zscore or False,
        )
        train_labels = self.train_dataset.dataset.targets
        self.train_ordered_labels = np.array(train_labels)

        self.model = self.create_encoder()
        self.viewmaker = self.create_viewmaker()
        self.random_viewmaker = (self.config.model_params.viewmaker_network == 'random')
        if self.random_viewmaker:
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        self.memory_bank = MemoryBank(
            len(self.train_dataset),
            self.config.model_params.out_dim,
        )

        if self.config.init_encoder_from_checkpoint not in [None, dotmap.DotMap()]:
            self.init_encoder_from_checkpoint()

    def get_t(self):
        if type(self.t) == int or type(self.t) == float:
            return self.t
        else:
            # If temperature is a free parameter, bound it to [0,1]
            return torch.sigmoid(self.t)

    def view(self, imgs):
        if 'Default' in self.config.system:
            print('Warning: calling self.view() with Default system')
            return imgs

        if self.config.data_params.normalize_before_view:
            imgs = self.normalize(imgs)

        views = self.viewmaker(imgs)

        if self.global_step % (len(self.train_dataset) // self.batch_size) == 0:
            imgs_to_log = imgs.permute(0,2,3,1).detach()[0].cpu().numpy()
            views_to_log = views.permute(0,2,3,1).detach()[0].cpu().numpy()
            views_to_log_conc = np.concatenate((imgs_to_log, views_to_log, views_to_log - imgs_to_log), axis=1)
            if self.train_dataset.NUM_CHANNELS > 3:
                for i in range(views_to_log_conc.shape[2]):
                    wandb.log({f"band {i} examples": wandb.Image(views_to_log_conc[:,:,i], caption=f"Epoch: {self.current_epoch}, Step {self.global_step}")})
            else:
                wandb.log({"examples": wandb.Image(views_to_log_conc, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}")})
        
        if not self.config.data_params.normalize_before_view:
            views = self.normalize(views)

        return views
        

    def view_k(self, imgs):
        # Momentum viewmaker for MoCo
        imgs = self.viewmaker_k(imgs)
        if 'Default' not in self.config.system:
            imgs = self.normalize(imgs)
        return imgs

    def create_encoder(self):
        if self.config.model_params.resnet_small:
            encoder_model = resnet_small.ResNet18(
                self.config.model_params.out_dim,
            )
            encoder_model.conv1 = nn.Conv2d(self.train_dataset.NUM_CHANNELS, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            resnet_class = getattr(
                torchvision.models, 
                self.config.model_params.resnet_version,
            )
            encoder_model = resnet_class(
                pretrained=False,
                num_classes=self.config.model_params.out_dim,
            )
            encoder_model.conv1 = nn.Conv2d(self.train_dataset.NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if self.config.model_params.projection_head:
            mlp_dim = encoder_model.fc.weight.size(1)
            encoder_model.fc = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                encoder_model.fc,
            )
        return encoder_model

    def init_encoder_from_checkpoint(self):
        if 'SimCLR' not in self.config.loss_params.name:
            # error out bc we not are saving the memory bank and we need
            assert NotImplementedError

        encoder_checkpoint = torch.load(self.config.init_encoder_from_checkpoint, map_location='cpu')
        encoder_state_dict = encoder_checkpoint['state_dict']
        new_encoder_state_dict = OrderedDict()
        for key, value in encoder_state_dict.items():
            # things are saved in DefaultSystem's encoder as model.x.x...
            # whereas in this encoder, things are saved as just x.x... so 
            # we need to remove the prefix
            if key == 'memory_bank._bank': 
                self.memory_bank._bank = value.clone()
            else:
                key = '.'.join(key.split('.')[1:])
                new_encoder_state_dict[key] = value
        self.model.load_state_dict(new_encoder_state_dict, strict=False)

    def create_viewmaker(self):
        filter_size = self.train_dataset.FILTER_SIZE
        view_model = Viewmaker(
            filter_size,
            self.config.model_params.noise_dim,
            device=self.device,
            num_channels=self.train_dataset.NUM_CHANNELS,
            L1_forced=self.config.model_params.view_L1_forced,
            bound_magnitude=self.config.model_params.view_bound_magnitude,
            divmaker=self.config.model_params.use_divmaker or False,
            activation=self.config.model_params.generator_activation or 'relu',
            clamp=self.config.model_params.clamp_views,
            symmetric_clamp=self.config.model_params.symmetric_clamp or False,
            num_res_blocks=self.config.model_params.num_res_blocks or 5
        )
        return view_model

    def noise(self, batch_size, device):
        shape = (batch_size, self.config.model_params.noise_dim)
        # Center noise at 0 then project to unit sphere.
        noise = utils.l2_normalize(torch.rand(shape, device=device) - 0.5)
        return noise
    
    def get_repr(self, img):
        if 'Default' not in self.config.system:
            img = self.normalize(img)
        return self.model(img)
    
    def normalize(self, imgs):
        if self.config.data_params.dataset == 'resisc45':
            mean = torch.tensor([0.377, 0.391, 0.363], device=imgs.device)
            std = torch.tensor([0.203, 0.189, 0.186], device=imgs.device)
        elif self.config.data_params.dataset == 'eurosat':
            mean = torch.tensor([1354.3003, 1117.7579, 1042.2800,  947.6443, 1199.6334, 2001.9829, 2372.5579, 2299.6663,  731.0175,   12.0956, 1822.4083, 1119.5759, 2598.4456], device=imgs.device)
            std = torch.tensor([244.0469, 323.4128, 385.0928, 584.1638, 566.0543, 858.5753, 1083.6704, 1103.0342, 402.9594, 4.7207, 1002.4071, 759.6080, 1228.4104], device=imgs.device)
        elif self.config.data_params.dataset == 'so2sat_sen1':
            mean = torch.tensor([-3.6247e-05, -7.5790e-06,  6.0370e-05,  2.5129e-05,  4.4201e-02, 2.5761e-01,  7.5741e-04,  1.3503e-03], device=imgs.device)
            std = torch.tensor([0.1756, 0.1756, 0.4600, 0.4560, 2.8554, 8.3233, 2.4494, 1.4644], device=imgs.device)
        elif self.config.data_params.dataset == 'so2sat_sen2':
            mean = torch.tensor([0.1238, 0.1093, 0.1011, 0.1142, 0.1593, 0.1815, 0.1746, 0.1950, 0.1543, 0.1091], device=imgs.device)
            std = torch.tensor([0.0396, 0.0478, 0.0664, 0.0636, 0.0774, 0.0910, 0.0922, 0.1016, 0.0999, 0.0878], device=imgs.device)
        elif self.config.data_params.dataset == 'bigearthnet':
            mean = torch.tensor([1562.0203, 1561.3035, 1562.4702, 1559.7567, 1560.7955, 1564.3341, 1558.2031, 1560.1460, 1563.4475, 1563.5408, 1559.8683, 1562.0842], device=imgs.device)
            std = torch.tensor([1635.5913, 1633.8538, 1634.5411, 1634.0165, 1636.4137, 1635.9663, 1634.6973, 1633.7421, 1634.8866, 1638.0367, 1635.1881, 1636.3383], device=imgs.device)
        else:
            raise ValueError(f'Dataset normalizer for {self.config.data_params.dataset} not implemented')
        imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
        return imgs

    def forward(self, batch, train=True):
        indices, img, img2, neg_img, _, = batch

        view1 = self.view(img)
        view2 = self.view(img2)

        emb_dict = {
            'indices': indices,
            'view1_embs': self.model(view1),
            'view2_embs': self.model(view2),
        }
        return emb_dict

    def get_losses_for_batch(self, emb_dict, train=True):
        view_maker_loss_weight = self.config.loss_params.view_maker_loss_weight
        if self.config.loss_params.objective != "AdversarialSimCLRLoss":
            raise Exception(f'Loss {self.config.loss_params.objective} is not supported or not using the pretrain system for model setup (should be adversarial simclr).')
        loss_function = AdversarialSimCLRLoss(
            embs1=emb_dict['view1_embs'],
            embs2=emb_dict['view2_embs'],
            t=self.get_t(),
            view_maker_loss_weight=view_maker_loss_weight
        )
        encoder_loss, view_maker_loss = loss_function.get_loss()
        img_embs = emb_dict['view1_embs'] 
        
        if self.random_viewmaker:
            view_maker_loss = self.dummy
        return encoder_loss, view_maker_loss

    def get_accuracies_for_batch(self, emb_dict):
        """
        Returns fraction of time that positives are closer than 
        negatives (for img-view and view-view).
        """
        if ('SimCLR' in self.loss_name):
            return -1, -1

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        img_view_sim = cos(emb_dict['img_embs'], emb_dict['pos_view_embs_2'])
        img_neg_sim = cos(emb_dict['img_embs'], emb_dict['neg_img_embs'])
        pos_pos_view_sim = cos(
            emb_dict['pos_view_embs_1'], emb_dict['pos_view_embs_2'])
        pos_neg_view_sim = cos(
            emb_dict['pos_view_embs_1'], emb_dict['neg_view_embs'])

        img_view_acc = (img_view_sim > img_neg_sim).float().mean()
        view_view_acc = (pos_pos_view_sim > pos_neg_view_sim).float().mean()

        return img_view_acc, view_view_acc

    def get_nearest_neighbor_label(self, img_embs, labels):
        """
        NOTE: ONLY TO BE USED FOR VALIDATION.

        For each image in validation, find the nearest image in the 
        training dataset using the memory bank. Assume its label as
        the predicted label.
        """
        batch_size = img_embs.size(0)
        all_dps = self.memory_bank.get_all_dot_products(img_embs)
        _, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
        neighbor_idxs = neighbor_idxs.squeeze(1)
        neighbor_idxs = neighbor_idxs.cpu().numpy()

        neighbor_labels = self.train_ordered_labels[neighbor_idxs]
        neighbor_labels = torch.from_numpy(neighbor_labels).long()

        if self.train_dataset.MULTI_LABEL or self.config.kappa_score:
            num_correct = torch.sum(neighbor_labels.cpu() == labels.cpu(), dim=0)
            num_correct = num_correct.detach().cpu().numpy()
            # return the actual labels for F1 score / Kappa score
            neighbor_labels = neighbor_labels.cpu()
            labels = labels.cpu()
            return num_correct, batch_size, neighbor_labels, labels
        else:
            num_correct = torch.sum(neighbor_labels.cpu() == labels.cpu()).item()
            return num_correct, batch_size

    def get_view_bound_magnitude(self):
        if self.config.model_params.view_bound_linear_scale:
            batch_size = self.config.optim_params.batch_size 
            num_epochs = self.config.num_epochs
            num_steps = int(math.ceil(len(self.train_dataset) / batch_size)) * num_epochs
            view_bound_end = self.config.model_params.view_bound_end
            view_bound_start = self.config.model_params.view_bound_start
            iter_incr = (view_bound_end - view_bound_start) / num_steps
            return view_bound_start + self.global_step * iter_incr
        else:
            return self.config.model_params.view_bound_magnitude  # constant
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        emb_dict = self.forward(batch)
        emb_dict['optimizer_idx'] = torch.tensor(optimizer_idx, device=self.device)
        encoder_loss, view_maker_loss = self.get_losses_for_batch(emb_dict, train=True)

        # Handle Tensor (dp) and int (ddp) cases
        if emb_dict['optimizer_idx'].__class__ == int or emb_dict['optimizer_idx'].dim() == 0:
            optimizer_idx = emb_dict['optimizer_idx'] 
        else:
            optimizer_idx = emb_dict['optimizer_idx'][0]
        if optimizer_idx == 0:
            metrics = {
                'encoder_loss': encoder_loss, 'temperature': self.get_t()
            }
            self.log("encoder_loss", encoder_loss)
            return {'loss': encoder_loss, 'log': metrics}
        else:
            # update the bound allowed for view
            self.viewmaker.bound_magnitude = self.get_view_bound_magnitude()

            metrics = {
                'view_maker_loss': view_maker_loss,
            }
            self.log('view_maker_loss', view_maker_loss)
            self.log('view_bound_magnitude', self.get_view_bound_magnitude())
            return {'loss': view_maker_loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        _, img, _, _, _ = batch
        img_embs = self.get_repr(img)  # Need encoding of image without augmentations (only normalization).
        labels = batch[-1]

        if self.train_dataset.MULTI_LABEL or self.config.kappa_score:
            num_correct, batch_size, pred_labels, true_labels = self.get_nearest_neighbor_label(img_embs, labels)
            output = OrderedDict({
                'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'val_num_total': torch.tensor(batch_size, dtype=float, device=self.device),
                'val_pred_labels': pred_labels.float().to(self.device),
                'val_true_labels': true_labels.float().to(self.device),
            })
        else:
            num_correct, batch_size = self.get_nearest_neighbor_label(img_embs, labels)
            output = OrderedDict({
                'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'val_num_total': torch.tensor(batch_size, dtype=float, device=self.device),
            })
        self.log('val_num_correct', torch.tensor(num_correct, dtype=float, device=self.device))
        self.log('val_num_total', torch.tensor(batch_size, dtype=float, device=self.device))
        return output

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            try:
                metrics[key] = torch.stack([elem[key] for elem in outputs]).mean()
            except:
                pass

        if self.train_dataset.MULTI_LABEL:
            num_class = self.train_dataset.NUM_CLASSES
            num_correct = torch.stack([out['val_num_correct'] for out in outputs], dim=1).sum(1)
            num_total = torch.stack([out['val_num_total'] for out in outputs]).sum()
            val_acc = num_correct / float(num_total)
            metrics['val_acc'] = val_acc.mean()
            self.log('val_acc', val_acc.mean())
            progress_bar = {'acc': val_acc.mean()}
            for c in range(num_class):
                val_acc_c = num_correct[c] / float(num_total)
                metrics[f'val_acc_feat{c}'] = val_acc_c
            # --- 
            val_pred_labels = torch.cat([out['val_pred_labels'] for out in outputs], dim=0).cpu().numpy()
            val_true_labels = torch.cat([out['val_true_labels'] for out in outputs], dim=0).cpu().numpy()

            # compute the f1 scores
            val_f1 = 0
            for c in range(num_class):
                val_f1_c = f1_score(val_true_labels[:, c], val_pred_labels[:, c])
                metrics[f'val_f1_feat{c}'] = val_f1_c
                val_f1 = val_f1 + val_f1_c
            val_f1 = val_f1 / float(num_class)
            metrics['val_f1'] = val_f1
            progress_bar['f1'] = val_f1
            return {'log': metrics, 
                    'val_acc': val_acc, 
                    'val_f1': val_f1, 
                    'progress_bar': progress_bar}
        elif self.config.kappa_score:
            num_correct = torch.stack([out['val_num_correct'] for out in outputs]).sum()
            num_total = torch.stack([out['val_num_total'] for out in outputs]).sum()
            val_acc = num_correct / float(num_total)

            val_pred_labels = torch.cat([out['val_pred_labels'] for out in outputs], dim=0).cpu().numpy()
            val_true_labels = torch.cat([out['val_true_labels'] for out in outputs], dim=0).cpu().numpy()

            val_kappa = cohen_kappa_score(
                val_true_labels, 
                val_pred_labels, 
                weights='quadratic',
            )
            metrics['val_acc'] = val_acc
            metrics['val_kappa'] = val_kappa
            progress_bar = {'acc': val_acc, 'kappa': val_kappa}
            self.log('val_acc', val_acc)
            self.log('val_kappa', val_kappa)
            return {'log': metrics, 
                    'val_acc': val_acc, 
                    'val_kappa': val_kappa, 
                    'progress_bar': progress_bar}
        else:
            num_correct = torch.stack([out['val_num_correct'] for out in outputs]).sum()
            num_total = torch.stack([out['val_num_total'] for out in outputs]).sum()
            val_acc = num_correct / float(num_total)
            metrics['val_acc'] = val_acc
            progress_bar = {'acc': val_acc}
            self.log('val_acc', val_acc)
            return {'log': metrics, 
                    'val_acc': val_acc, 
                    'progress_bar': progress_bar}

    def configure_optimizers(self):
        # Optimize temperature with encoder.
        if type(self.t) == float or type(self.t) == int:
            encoder_params = self.model.parameters()
        else:
            encoder_params = list(self.model.parameters()) + [self.t]

        encoder_optim = torch.optim.SGD(
            encoder_params,
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
        )
        view_optim_name = self.config.optim_params.viewmaker_optim
        view_parameters = self.viewmaker.parameters()
        if view_optim_name == 'adam':
            view_optim = torch.optim.Adam(
                view_parameters, lr=self.config.optim_params.viewmaker_learning_rate or 0.001)
        elif not view_optim_name or view_optim_name == 'sgd':
            view_optim = torch.optim.SGD(
                view_parameters,
                lr=self.config.optim_params.viewmaker_learning_rate or self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        else:
            raise ValueError(f'Optimizer {view_optim_name} not implemented')
        
        if self.random_viewmaker:
            view_optim = torch.optim.Adam([self.dummy])

        return [encoder_optim, view_optim], []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size, 
                                 shuffle=False, drop_last=False)


class PretrainDivMakerSystem(PretrainSystem):
    """
    DivMaker Pretraining.
    """

    def __init__(self, config):
        super().__init__(config)
        if (self.config.loss_params.objective == "DivMakerLoss" and self.config.model_params.use_divmaker):
            print("Using DivMaker loss")
        else:
            raise Exception(f'Loss {self.config.loss_params.objective} is not supported or not using DivMaker for model setup.')

        self.t = self.config.loss_params.t
        self.automatic_optimization = False  # This tells Lightning to let us control the training loop.
        torch.autograd.set_detect_anomaly(True)

    def forward(self, img, view=True, train=True):
        if not view:
            embs = self.model(img)
            return embs

        views = self.view(img)
        embs = self.model(views)
        return embs

    def get_enc_loss(self, imgs):
        embs1 = self.forward(imgs)
        embs2 = self.forward(imgs)
        loss_function = SimCLRObjective(embs1, embs2, t=self.t)
        encoder_loss= loss_function.get_loss()
        return encoder_loss

    def get_vm_loss(self, imgs):
        embs = self.forward(imgs, view=False)
        embs1 = self.forward(imgs)
        embs2 = self.forward(imgs)
        if self.config.loss_params.num_views == 2:
            loss_function = DivMakerLoss(embs, embs1, embs2, t=self.t)
        elif self.config.loss_params.num_views == 3:
            embs3 = self.forward(imgs)
            loss_function = DivMakerLoss(embs, embs1, embs2, embs3, t=self.t)
        else:
            raise ValueError(f'Only 2 or 3 views are supported for DivMaker Loss.')
        vm_loss = loss_function.get_loss()
        return vm_loss

    def get_view_bound_magnitude(self):
        return self.config.model_params.view_bound_magnitude

    def training_step(self, batch, batch_idx):
        indices, img, img2, neg_img, _, = batch
        encoder_loss = self.get_enc_loss(img)
        encoder_optim, vm_optim = self.optimizers()
        self.manual_backward(encoder_loss)

        # alternate optimization steps between encoder and viewmaker.
        encoder_optim.step()
        encoder_optim.zero_grad()
        vm_optim.zero_grad()

        # compute loss for the divmaker
        vm_loss = self.get_vm_loss(img)
        self.manual_backward(vm_loss) 
        vm_optim.step()
        vm_optim.zero_grad()
        encoder_optim.zero_grad()

        metrics = {
            'view_maker_loss': vm_loss,
            'encoder_loss': encoder_loss, 
            'temperature': self.t,
        }
        self.log('encoder_loss', encoder_loss)
        self.log('view_maker_loss', vm_loss)
        return {'loss': vm_loss, 'enc_loss': encoder_loss, 'log': metrics}


class TruncatedDataset(torch.utils.data.Sampler):
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return iter(list(range(self.n)))

    def __len__(self):
        return self.n 


def create_dataloader(dataset, config, batch_size, shuffle=True, drop_last=True):
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        # shuffle=False, 
        pin_memory=True,
        drop_last=drop_last,
        num_workers=config.data_loader_workers,
        # sampler=TruncatedDataset(n=1000),
    )
    return loader


# ----- Transfer Systems -----

class TransferSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.encoder, self.viewmaker, self.system, self.pretrain_config = self.load_pretrained_model()
        resnet = self.pretrain_config.model_params.resnet_version
        
        if resnet == 'resnet18':
            if self.config.model_params.use_prepool:
                if self.pretrain_config.model_params.resnet_small:
                    num_features = 512 * 4 * 4
                else:
                    num_features = 512 * 7 * 7
            else:
                num_features = 512
        elif resnet == 'resnet34':
            num_features = 512
        elif resnet == 'resnet50':
            num_features = 2048
        else:
            raise Exception(f'resnet {resnet} not supported.')
        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
            default_augmentations=self.pretrain_config.data_params.default_augmentations or False,
            resize_imagenet_to_32=self.pretrain_config.data_params.resize_imagenet_to_32 or False,
            mask=False,  # Don't mask during transfer.
            zscore=self.pretrain_config.data_params.zscore or False,
        )
        if not self.pretrain_config.model_params.resnet_small:
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # keep pooling layer

        self.should_finetune = self.config.model_params.finetune or False
        if self.should_finetune:
            self.encoder = self.encoder.train()
            self.viewmaker = self.viewmaker.eval()
            utils.free_params(self.encoder)
            utils.frozen_params(self.viewmaker)
        else:
            self.encoder = self.encoder.eval()
            self.viewmaker = self.viewmaker.eval()
            utils.frozen_params(self.encoder)
            utils.frozen_params(self.viewmaker)

        self.num_features = num_features
        self.model = self.create_model()

    def load_pretrained_model(self):
        base_dir = self.config.pretrain_model.exp_dir
        checkpoint_name = self.config.pretrain_model.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)
        
        if self.config.model_params.resnet_small:
            config.model_params.resnet_small = self.config.model_params.resnet_small

        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        system.load_state_dict(checkpoint['state_dict'], strict=False)

        encoder = system.model.eval()
        viewmaker = system.viewmaker.eval()

        return encoder, viewmaker, system, system.config

    def create_model(self):
        num_class = self.train_dataset.NUM_CLASSES
        model = LogisticRegression(self.num_features, num_class)
        return model

    def noise(self, batch_size):
        shape = (batch_size, self.pretrain_config.model_params.noise_dim)
        # Center noise at 0 then project to unit sphere.
        noise = utils.l2_normalize(torch.rand(shape) - 0.5)
        return noise

    def forward(self, img, valid=False):
        batch_size = img.size(0)
        if self.config.data_params.normalize_before_view:
            img = self.system.normalize(img)
        if not valid and not self.config.optim_params.no_views: 
            img = self.viewmaker(img)
            if type(img) == tuple:
                idx = random.randint(0, 1)
                img = img[idx]
        if 'Default' not in self.pretrain_config.system and not self.config.data_params.normalize_before_view:
            img = self.system.normalize(img)
        if self.pretrain_config.model_params.resnet_small:
            if self.config.model_params.use_prepool:
                embs = self.encoder(img, layer=5)
            else:
                embs = self.encoder(img, layer=6)
        else:
            embs = self.encoder(img)
        return self.model(embs.view(batch_size, -1))

    def get_losses_for_batch(self, batch, valid=False):
        _, img, _, _, label = batch
        logits = self.forward(img, valid)
        if self.train_dataset.MULTI_LABEL:
            return F.binary_cross_entropy(torch.sigmoid(logits).view(-1), 
                                          label.view(-1).float())
        else:
            return F.cross_entropy(logits, label.long())

    def get_accuracies_for_batch(self, batch, valid=False):
        _, img, _, _, label = batch
        batch_size = img.size(0)
        logits = self.forward(img, valid)
        if self.train_dataset.MULTI_LABEL:
            preds = torch.round(torch.sigmoid(logits))
            preds = preds.long().cpu()
            num_correct = torch.sum(preds.cpu() == label.cpu(), dim=0)
            num_correct = num_correct.detach().cpu().numpy()
            num_total = batch_size
            return num_correct, num_total, preds, label.cpu()
        else:
            preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            preds = preds.long().cpu()
            num_correct = torch.sum(preds == label.long().cpu()).item()
            num_total = batch_size
            return num_correct, num_total

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        with torch.no_grad():
            if self.train_dataset.MULTI_LABEL:
                num_correct, num_total, _, _ = self.get_accuracies_for_batch(batch)
                num_correct = num_correct.mean()
            else:
                num_correct, num_total = self.get_accuracies_for_batch(batch)
            metrics = {
                'train_loss': loss,
                'train_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'train_num_total': torch.tensor(num_total, dtype=float, device=self.device),
                'train_acc': torch.tensor(num_correct / float(num_total), dtype=float, device=self.device)
            }
        self.log('train_loss', loss.item())
        self.log('train_num_correct', num_correct)
        self.log('train_num_total', num_total)
        self.log('train_acc', num_correct / float(num_total))
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, valid=True)
        if self.train_dataset.MULTI_LABEL:  # regardless if binary or not
            num_correct, num_total, val_preds, val_labels = \
                self.get_accuracies_for_batch(batch, valid=True)
            self.log('val_loss', loss.item())
            self.log('val_num_correct', num_correct.mean())
            self.log('val_num_total', num_total)
            self.log('val_acc', num_correct.mean() / float(num_total))
            return OrderedDict({
                'val_loss': loss,
                'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'val_num_total': torch.tensor(num_total, dtype=float, device=self.device),
                'val_acc': torch.tensor(num_correct / float(num_total), dtype=float, device=self.device),
                'val_pred_labels': val_preds.float(),
                'val_true_labels': val_labels.float(),
            })
        else:
            num_correct, num_total = self.get_accuracies_for_batch(batch, valid=True)
            self.log('val_loss', loss.item())
            self.log('val_num_correct', num_correct)
            self.log('val_num_total', num_total)
            self.log('val_acc', num_correct / float(num_total))
            return OrderedDict({
                'val_loss': loss,
                'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'val_num_total': torch.tensor(num_total, dtype=float, device=self.device),
                'val_acc': torch.tensor(num_correct / float(num_total), dtype=float, device=self.device),
            })

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            try:
                metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
            except:
                pass
        
        if self.train_dataset.MULTI_LABEL:
            num_correct = torch.stack([out['val_num_correct'] for out in outputs], dim=1).sum(1)
            num_total = torch.stack([out['val_num_total'] for out in outputs]).sum()
            val_acc = num_correct / float(num_total)
            metrics['val_acc'] = val_acc.mean()
            progress_bar = {'acc': val_acc.mean()}
            num_class = self.train_dataset.NUM_CLASSES
            for c in range(num_class):
                val_acc_c = num_correct[c] / float(num_total)
                metrics[f'val_acc_feat{c}'] = val_acc_c
            val_pred_labels = torch.cat([out['val_pred_labels'] for out in outputs], dim=0).numpy()
            val_true_labels = torch.cat([out['val_true_labels'] for out in outputs], dim=0).numpy()
        
            val_f1 = 0
            val_f2 = 0
            val_precision = 0
            val_recall = 0
            for c in range(num_class):
                val_f1_c = f1_score(val_true_labels[:, c], val_pred_labels[:, c])
                val_f2_c = fbeta_score(val_true_labels[:, c], val_pred_labels[:, c], beta=2)
                val_precision_c = precision_score(val_true_labels[:, c], val_pred_labels[:, c])
                val_recall_c = recall_score(val_true_labels[:, c], val_pred_labels[:, c])
                metrics[f'val_f1_feat{c}'] = val_f1_c
                val_f1 = val_f1 + val_f1_c
                val_f2 = val_f2 + val_f2_c
                val_precision = val_precision + val_precision_c
                val_recall = val_recall + val_recall_c
            val_f1 = val_f1 / float(num_class)
            val_f2 = val_f2 / float(num_class)
            val_precision = val_precision / float(num_class)
            val_recall = val_recall / float(num_class)
            metrics['val_f1'] = val_f1
            progress_bar['f1'] = val_f1
            self.log('val_acc', val_acc)
            self.log('val_f1', val_f1)
            self.log('val_f2', val_f2)
            self.log('val_precision', val_precision)
            self.log('val_recall', val_recall)
            return {'val_loss': metrics['val_loss'], 
                    'log': metrics,
                    'val_acc': val_acc, 
                    'val_f1': val_f1,
                    'progress_bar': progress_bar}
        elif self.pretrain_config.kappa_score:
            num_correct = sum([out['val_num_correct'] for out in outputs])
            num_total = sum([out['val_num_total'] for out in outputs])
            val_acc = num_correct / float(num_total)
            val_kappa = cohen_kappa_score(
                val_true_labels, 
                val_pred_labels, 
                weights='quadratic',
            )
            metrics['val_acc'] = val_acc
            metrics['val_kappa'] = val_kappa
            progress_bar = {'acc': val_acc, 'kappa': val_kappa}
            self.log('val_acc', val_acc)
            return {'val_loss': metrics['val_loss'], 
                    'log': metrics, 
                    'val_acc': val_acc, 
                    'val_kappa': val_kappa, 
                    'progress_bar': progress_bar}
        else:
            num_correct = sum([out['val_num_correct'] for out in outputs])
            num_total = sum([out['val_num_total'] for out in outputs])
            val_acc = num_correct / float(num_total)
            metrics['val_acc'] = val_acc
            progress_bar = {'acc': val_acc}
            self.log('val_acc', val_acc)
            return {'val_loss': metrics['val_loss'], 
                    'log': metrics, 
                    'val_acc': val_acc,
                    'progress_bar': progress_bar}

    def configure_optimizers(self):
        if self.should_finetune:
            params_iterator = chain(self.model.parameters(), self.encoder.parameters())
        else:
            params_iterator = self.model.parameters()

        if self.config.optim_params == 'adam':
            optim = torch.optim.Adam(params_iterator)
        else:
            optim = torch.optim.SGD(
                params_iterator,
                lr=self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        return [optim], []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size, 
                                 shuffle=False, drop_last=False)


# ----- Baseline Systems -----

class LinearSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
        )
        assert not self.train_dataset.MULTI_LABEL, "class does not support multilabel"
        self.model = self.create_model()

    def create_model(self):
        num_class = self.train_dataset.NUM_CLASSES
        model = LogisticRegression(3*32*32, num_class)
        return model

    def forward(self, img):
        batch_size = img.size(0)
        img = img.view(batch_size, -1)
        return self.model(img)

    def get_losses_for_batch(self, batch):
        _, img, _, _, label = batch
        logits = self.forward(img)
        return F.cross_entropy(logits, label)

    def get_accuracies_for_batch(self, batch):
        _, img, _, _, label = batch
        logits = self.forward(img)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        preds = preds.long().cpu()
        num_correct = torch.sum(preds == label.long().cpu()).item()
        num_total = img.size(0)

        return num_correct, num_total

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        with torch.no_grad():
            num_correct, num_total = self.get_accuracies_for_batch(batch)
            metrics = {
                'train_loss': loss,
                'train_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'train_num_total': torch.tensor(num_total, dtype=float, device=self.device),
                'train_acc': torch.tensor(num_correct / float(num_total), dtype=float, device=self.device),
            }
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        num_correct, num_total = self.get_accuracies_for_batch(batch)
        return OrderedDict({
            'val_loss': loss,
            'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
            'val_num_total': torch.tensor(num_total, dtype=float, device=self.device),
            'val_acc': torch.tensor(num_correct / float(num_total), dtype=float, device=self.device),
        })

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
        num_correct = sum([out['val_num_correct'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        return {'val_loss': metrics['val_loss'], 'log': metrics, 'val_acc': val_acc}

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
        )
        return [optim], []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size, 
                                 shuffle=False, drop_last=False)


class DefaultSystem(PretrainSystem):

    def __init__(self, config):
        super(PretrainSystem, self).__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.loss_name = self.config.loss_params.name

        # Optionally optimize temperature.
        if self.config.loss_params.optim_t:
            self.t = torch.nn.Parameter(torch.tensor(self.config.loss_params.t), requires_grad=True)
        else:
            self.t = self.config.loss_params.t

        default_augmentations = self.config.data_params.default_augmentations
        if default_augmentations == DotMap():
           default_augmentations = 'all'
        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
            default_augmentations=default_augmentations,
            resize_imagenet_to_32=self.config.data_params.resize_imagenet_to_32 or False,
            mask=self.config.data_params.mask or False,
            zscore=self.config.data_params.zscore or False,
        )
        train_labels = self.train_dataset.dataset.targets
        self.train_ordered_labels = np.array(train_labels)
        self.model = self.create_encoder()

        self.memory_bank = MemoryBank(
            len(self.train_dataset), 
            self.config.model_params.out_dim, 
        )

    def forward(self, img):
        if self.global_step == 5:
            views_to_log = img.permute(0,2,3,1).detach()[0].cpu().numpy()
            wandb.log({"examples": [wandb.Image(view, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}") for view in views_to_log]})
        return self.model(img)

    def get_losses_for_batch(self, emb_dict, train=True):
        if self.loss_name == 'simclr':
            if 'img_embs_2' not in emb_dict:
                raise ValueError(f'img_embs_2 is required for SimCLR loss')
            loss_fn = SimCLRObjective(emb_dict['img_embs_1'], emb_dict['img_embs_2'],
                                      t=self.get_t(),
                                      push_only=self.config.loss_params.push_only)
            loss = loss_fn.get_loss()
        else:
            raise Exception(f'Objective {self.loss_name} is not supported.')

        if train:
            with torch.no_grad():
                if 'simclr' in self.loss_name:
                    outputs_avg = (utils.l2_normalize(emb_dict['img_embs_1'], dim=1) + 
                                   utils.l2_normalize(emb_dict['img_embs_2'], dim=1)) / 2.
                    indices = emb_dict['indices']
                    self.memory_bank.update(indices, outputs_avg)
                else:
                    raise Exception(f'Objective {self.loss_name} is not supported.')

        return loss

    def configure_optimizers(self):
        # Optimize temperature with encoder.
        if type(self.t) == float or type(self.t) == int:
            encoder_params = self.model.parameters()
        else:
            encoder_params = list(self.model.parameters()) + [self.t]

        if self.config.optim_params.adam:
            optim = torch.optim.AdamW(encoder_params)
        else:
            optim = torch.optim.SGD(
                encoder_params,
                lr=self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        return [optim], []

    def training_step(self, batch, batch_idx):
        emb_dict = {}
        indices, img, img2, neg_img, labels, = batch
        if 'simclr' in self.loss_name:
            emb_dict['img_embs_1'] = self.forward(img)
            emb_dict['img_embs_2'] = self.forward(img2)

        emb_dict['indices'] = indices
        emb_dict['labels'] = labels
        return emb_dict

    def training_step_end(self, emb_dict):
        loss = self.get_losses_for_batch(emb_dict, train=True)
        metrics = {'loss': loss, 'temperature': self.get_t()}
        return {'loss': loss, 'log': metrics}
    
    def validation_step(self, batch, batch_idx):
        emb_dict = {}
        indices, img, img2, neg_img, labels, = batch
        if 'simclr' in self.loss_name:
            emb_dict['img_embs_1'] = self.forward(img)
            emb_dict['img_embs_2'] = self.forward(img2)
        
        emb_dict['indices'] = indices
        emb_dict['labels'] = labels
        img_embs = emb_dict['img_embs_1']
        
        loss = self.get_losses_for_batch(emb_dict, train=False)

        if self.train_dataset.MULTI_LABEL or self.config.kappa_score:
            num_correct, batch_size, pred_labels, true_labels = \
                self.get_nearest_neighbor_label(img_embs, labels)
            output = OrderedDict({
                'val_loss': loss,
                'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'val_num_total': torch.tensor(batch_size, dtype=float, device=self.device),
                'val_pred_labels': pred_labels.float(),
                'val_true_labels': true_labels.float(),
            })
        else:
            num_correct, batch_size = self.get_nearest_neighbor_label(img_embs, labels)
            output = OrderedDict({
                'val_loss': loss,
                'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'val_num_total': torch.tensor(batch_size, dtype=float, device=self.device),
            })
        return output


class TransferDefaultSystem(TransferSystem):

    def __init__(self, config):
        super(TransferSystem, self).__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        
        self.encoder, self.pretrain_config = self.load_pretrained_model()
        resnet = self.pretrain_config.model_params.resnet_version
        if resnet == 'resnet18':
            if self.config.model_params.use_prepool:
                if self.pretrain_config.model_params.resnet_small:
                    num_features = 512 * 4 * 4
                else:
                    num_features = 512 * 7 * 7
            else:
                num_features = 512
        elif resnet == 'resnet34':
            num_features = 512
        elif resnet == 'resnet50':
            num_features = 2048
        else:
            raise Exception(f'resnet {resnet} not supported.')

        if not self.pretrain_config.model_params.resnet_small:
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # keep pooling layer
        
        self.should_finetune = self.config.model_params.finetune or False
        if self.should_finetune:
            self.encoder = self.encoder.train()
            utils.free_params(self.encoder)
        else:
            self.encoder = self.encoder.eval()
            utils.frozen_params(self.encoder)

        default_augmentations = self.pretrain_config.data_params.default_augmentations
        if self.config.data_params.force_default_views:
            default_augmentations = 'all'
        if default_augmentations == DotMap():
           default_augmentations = 'all'
        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
            default_augmentations=default_augmentations,
            resize_imagenet_to_32=self.pretrain_config.data_params.resize_imagenet_to_32 or False,
            mask=False,  # Don't mask during transfer.
            zscore=self.config.data_params.zscore or False,
        )
        self.num_features = num_features
        self.model = self.create_model()

    def load_pretrained_model(self):
        base_dir = self.config.pretrain_model.exp_dir
        checkpoint_name = self.config.pretrain_model.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)

        if self.config.model_params.resnet_small:
            config.model_params.resnet_small = self.config.model_params.resnet_small

        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        system.load_state_dict(checkpoint['state_dict'], strict=False)

        encoder = system.model.eval()
        return encoder, config

    def forward(self, img, unused_valid=None):
        del unused_valid
        batch_size = img.size(0)
        if self.pretrain_config.model_params.resnet_small:
            if self.config.model_params.use_prepool:
                embs = self.encoder(img, layer=5)
            else:
                embs = self.encoder(img, layer=6)
        else:
            embs = self.encoder(img)
        return self.model(embs.view(batch_size, -1))


class TransferBigEarthNetSystem(TransferSystem):
    # Copy of default transfer system except we compute F1.

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, valid=True)
        num_correct, num_total, val_preds, val_labels = \
            self.get_accuracies_for_batch(batch, valid=True)
        self.log('val_acc', (num_correct / float(num_total)).mean())
        return OrderedDict({
            'val_loss': loss,
            'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
            'val_num_total': torch.tensor(num_total, dtype=float, device=self.device),
            'val_acc': torch.tensor(num_correct / float(num_total), dtype=float, device=self.device),
            'val_pred_labels': val_preds.float(),
            'val_true_labels': val_labels.float(),
        })

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            try:
                metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
            except:
                pass

        num_correct = torch.stack([out['val_num_correct'] for out in outputs], dim=1).sum(1)
        num_total = torch.stack([out['val_num_total'] for out in outputs]).sum()
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc.mean()
        progress_bar = {'acc': val_acc.mean()}
        num_class = self.train_dataset.NUM_CLASSES
        for c in range(num_class):
            val_acc_c = num_correct[c] / float(num_total)
            metrics[f'val_acc_feat{c}'] = val_acc_c
        val_pred_labels = torch.cat([out['val_pred_labels'] for out in outputs], dim=0).numpy()
        val_true_labels = torch.cat([out['val_true_labels'] for out in outputs], dim=0).numpy()

        val_f1 = 0
        val_f2 = 0
        val_precision = 0
        val_recall = 0
        for c in range(num_class):
            val_f1_c = f1_score(val_true_labels[:, c], val_pred_labels[:, c])
            val_f2_c = fbeta_score(val_true_labels[:, c], val_pred_labels[:, c], beta=2)
            val_precision_c = precision_score(val_true_labels[:, c], val_pred_labels[:, c])
            val_recall_c = recall_score(val_true_labels[:, c], val_pred_labels[:, c])
            metrics[f'val_f1_feat{c}'] = val_f1_c
            val_f1 = val_f1 + val_f1_c
            val_f2 = val_f2 + val_f2_c
            val_precision = val_precision + val_precision_c
            val_recall = val_recall + val_recall_c
        val_f1 = val_f1 / float(num_class)
        val_f2 = val_f2 / float(num_class)
        val_precision = val_precision / float(num_class)
        val_recall = val_recall / float(num_class)
        metrics['val_f1'] = val_f1
        progress_bar['f1'] = val_f1
        self.log('val_acc', val_acc.mean())
        self.log('val_f1', val_f1.mean())
        self.log('val_f2', val_f2.mean())
        self.log('val_precision', val_precision.mean())
        self.log('val_recall', val_recall.mean())
        return {'val_loss': metrics['val_loss'], 
                'log': metrics,
                'val_acc': val_acc, 
                'val_f1': val_f1,
                'progress_bar': progress_bar}


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
