"""
Trainable Backbone + MLP Classifier + LwF (Learning without Forgetting)
Inherits from MLP baseline and adds LwF functionality
"""

import torch
import torch.nn.functional as F
from copy import deepcopy
from argparse import Namespace

from models.trainable_backbone_mlp_pretrained import TrainableBackboneMLPPretrained
from utils.args import ArgumentParser


class TrainableBackboneMLPLwF(TrainableBackboneMLPPretrained):
    """Trainable backbone with MLP classifier and LwF"""
    NAME = 'trainable_backbone_mlp_lwf_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone: torch.nn.Module, loss: torch.nn.Module,
                 args: Namespace, transform: torch.nn.Module, dataset):
        
        # Call MLP parent to get MLP classifier + differential LR setup
        super().__init__(backbone, loss, args, transform, dataset)
        
        # Add LwF-specific attributes
        self.old_net = None
        self.eye = torch.eye(self.num_classes).to(self.device)
        
        print(f"\n{'='*70}")
        print(f"[Trainable-LwF-MLP] Learning without Forgetting enabled")
        print(f"[Trainable-LwF-MLP] Classifier type: {type(self.net.classifier)}")
        print(f"{'='*70}\n")
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """Training with LwF distillation"""
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
        """Save model copy after each task"""
        print(f"\n[LwF] Saving model copy after Task {self.current_task + 1}...")
        
        self.old_net = deepcopy(self.net)
        self.old_net.eval()
        
        for param in self.old_net.parameters():
            param.requires_grad = False
        
        self.net.train()
        
        print(f"[LwF] ✓ Model copy saved and frozen\n")