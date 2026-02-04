"""
Frozen Backbone + Linear Classifier + LwF
Based on Mammoth's lwf_mc but adapted for frozen backbone
"""

import torch
import torch.nn.functional as F
from copy import deepcopy
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace


class FrozenBackboneLinearLwF(ContinualModel):
    """Frozen backbone with Linear classifier and LwF"""
    NAME = 'frozen_backbone_linear_lwf_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone: torch.nn.Module, loss: torch.nn.Module,
                 args: Namespace, transform: torch.nn.Module, dataset: ContinualDataset):
        
        super().__init__(backbone, loss, args, transform, dataset)
        
        # Freeze backbone
        self._freeze_backbone()
        self._verify_frozen()
        
        # LwF: Store old model
        self.old_net = None
        
        # Create eye matrix for binary cross-entropy
        self.eye = torch.eye(self.num_classes).to(self.device)
        
        print(f"\n{'='*70}")
        print(f"[LwF-Linear] Learning without Forgetting enabled")
        print(f"[LwF-Linear] Using binary cross-entropy (multi-label)")
        print(f"{'='*70}\n")
    
    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        print("\n" + "="*70)
        print("[FREEZE] Freezing backbone...")
        print("="*70)
        
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.net.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
                trainable_count += param.numel()
        
        print(f"[FREEZE] ✓ Frozen: {frozen_count:,}")
        print(f"[FREEZE] ✓ Trainable: {trainable_count:,}")
        print("="*70 + "\n")
    
    def _verify_frozen(self):
        """Verify freeze status"""
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        frozen = total - trainable
        
        print(f"[VERIFY] Total: {total:,}, Trainable: {trainable:,}")
        print(f"[VERIFY] Frozen: {100 * frozen / total:.1f}%\n")
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        Training with LwF distillation.
        Based on Mammoth's lwf_mc implementation.
        """
        # Get old model predictions (if available)
        if self.current_task > 0 and self.old_net is not None:
            with torch.no_grad():
                old_logits = torch.sigmoid(self.old_net(inputs))
        else:
            old_logits = None
        
        # Zero gradients
        self.opt.zero_grad()
        
        # Compute loss
        loss = self.get_loss(inputs, labels, self.current_task, old_logits)
        
        # Backward and optimize
        loss.backward()
        self.opt.step()
        
        return loss.item()
    
    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, old_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute LwF loss.
        
        Uses binary cross-entropy to allow simultaneous learning of:
        - New classes (from current data)
        - Old classes (from distillation)
        """
        # Calculate class range for current task
        classes_per_task = self.num_classes // self.n_tasks
        pc = task_idx * classes_per_task  # Past classes
        ac = (task_idx + 1) * classes_per_task  # All classes up to now
        
        # Forward pass
        outputs = self.net(inputs)[:, :ac]
        
        if task_idx == 0:
            # First task: standard cross-entropy
            targets = self.eye[labels][:, :ac]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
        else:
            # Subsequent tasks: distillation + new classes
            # New classes targets
            targets = self.eye[labels][:, pc:ac]
            
            # Combined: old predictions (from teacher) + new targets
            comb_targets = torch.cat((old_logits[:, :pc], targets), dim=1)
            
            # Binary cross-entropy on combined targets
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
        
        return loss
    
    def end_task(self, dataset):
        """Save model copy after each task"""
        print(f"\n{'='*70}")
        print(f"[LwF] Saving model copy after Task {self.current_task + 1}...")
        print(f"{'='*70}")
        
        # Deep copy of current model
        self.old_net = deepcopy(self.net)
        self.old_net.eval()
        
        # Freeze old model
        for param in self.old_net.parameters():
            param.requires_grad = False
        
        # Keep current model in training mode
        self.net.train()
        
        print(f"[LwF] ✓ Model copy saved and frozen")
        print(f"{'='*70}\n")
    
    @staticmethod
    def get_parser(parser):
        parser.add_argument('--wd_reg', type=float, default=0.0,
                          help='L2 regularization (default: 0.0)')
        return parser