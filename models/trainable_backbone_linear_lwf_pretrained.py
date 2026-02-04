"""
Trainable Backbone + Linear Classifier + LwF (Learning without Forgetting)
Uses differential learning rates for backbone vs classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from argparse import Namespace

from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from utils.args import ArgumentParser


class TrainableBackboneLinearLwF(ContinualModel):
    """Trainable backbone with Linear classifier and LwF"""
    NAME = 'trainable_backbone_linear_lwf_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone: torch.nn.Module, loss: torch.nn.Module,
                 args: Namespace, transform: torch.nn.Module, dataset: ContinualDataset):
        
        super().__init__(backbone, loss, args, transform, dataset)
        
        # All parameters trainable
        self._set_all_trainable()
        self._verify_trainable()
        self._print_trainable_summary()
        
        # Setup differential learning rates
        self._setup_differential_lr()
        
        # LwF setup
        self.old_net = None
        self.eye = torch.eye(self.num_classes).to(self.device)
        
        print(f"\n{'='*70}")
        print(f"[Trainable-LwF-Linear] LwF with trainable backbone")
        print(f"{'='*70}\n")
    
    def _set_all_trainable(self):
        """Ensure all parameters are trainable"""
        print("\n" + "="*70)
        print("[TRAINABLE] Setting all parameters to trainable...")
        print("="*70)
        
        trainable_count = 0
        for param in self.net.parameters():
            param.requires_grad = True
            trainable_count += param.numel()
        
        print(f"[TRAINABLE] ✓ All parameters trainable: {trainable_count:,}")
        print("="*70 + "\n")
    
    def _verify_trainable(self):
        """Verify all parameters are trainable"""
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        
        print(f"[VERIFY] Total: {total:,}, Trainable: {trainable:,}")
        print(f"[VERIFY] All trainable: {trainable == total}\n")
    
    def _print_trainable_summary(self):
        """Print parameter groups"""
        backbone_params = 0
        classifier_params = 0
        
        for name, param in self.net.named_parameters():
            if 'classifier' in name:
                classifier_params += param.numel()
            else:
                backbone_params += param.numel()
        
        print(f"[SUMMARY] Backbone: {backbone_params:,}, Classifier: {classifier_params:,}\n")
    
    def _setup_differential_lr(self):
        """Setup optimizer with differential learning rates"""
        backbone_lr = getattr(self.args, 'backbone_lr', self.args.lr * 0.1)
        classifier_lr = self.args.lr
        
        print("\n" + "="*70)
        print("[OPTIMIZER] Setting up differential learning rates...")
        print("="*70)
        print(f"[OPTIMIZER] Backbone LR: {backbone_lr:.6f}")
        print(f"[OPTIMIZER] Classifier LR: {classifier_lr:.6f}")
        print(f"[OPTIMIZER] Ratio: {classifier_lr/backbone_lr:.1f}x")
        print("="*70 + "\n")
        
        backbone_params = []
        classifier_params = []
        
        for name, param in self.net.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': classifier_params, 'lr': classifier_lr}
        ]
        
        if self.args.optimizer == 'sgd':
            self.opt = torch.optim.SGD(
                param_groups,
                momentum=self.args.optim_mom,
                weight_decay=self.args.optim_wd,
                nesterov=self.args.optim_nesterov
            )
        elif self.args.optimizer == 'adam':
            self.opt = torch.optim.Adam(
                param_groups,
                weight_decay=self.args.optim_wd
            )
        elif self.args.optimizer == 'adamw':
            self.opt = torch.optim.AdamW(
                param_groups,
                weight_decay=self.args.optim_wd
            )
        
        print(f"[OPTIMIZER] ✓ Created with {len(param_groups)} parameter groups\n")
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """Training with LwF"""
        # Get old model predictions (if available)
        if self.current_task > 0 and self.old_net is not None:
            with torch.no_grad():
                old_logits = torch.sigmoid(self.old_net(inputs))
        else:
            old_logits = None
        
        self.opt.zero_grad()
        loss = self.get_loss(inputs, labels, self.current_task, old_logits)
        loss.backward()
        self.opt.step()
        
        return loss.item()
    
    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, old_logits: torch.Tensor) -> torch.Tensor:
        """Compute LwF loss"""
        classes_per_task = self.num_classes // self.n_tasks
        pc = task_idx * classes_per_task
        ac = (task_idx + 1) * classes_per_task
        
        outputs = self.net(inputs)[:, :ac]
        
        if task_idx == 0:
            targets = self.eye[labels][:, :ac]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
        else:
            targets = self.eye[labels][:, pc:ac]
            comb_targets = torch.cat((old_logits[:, :pc], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
        
        return loss
    
    def end_task(self, dataset):
        """Save model copy"""
        print(f"\n{'='*70}")
        print(f"[LwF] Saving model copy after Task {self.current_task + 1}...")
        print(f"{'='*70}")
        
        self.old_net = deepcopy(self.net)
        self.old_net.eval()
        
        for param in self.old_net.parameters():
            param.requires_grad = False
        
        self.net.train()
        
        print(f"[LwF] ✓ Model copy saved and frozen")
        print(f"{'='*70}\n")
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--backbone_lr', type=float, default=None,
                          help='Learning rate for backbone (default: 0.1 * lr)')
        parser.add_argument('--wd_reg', type=float, default=0.0,
                          help='L2 regularization (default: 0.0)')
        return parser