"""
Linear Classifier with MAS - Frozen Backbone (FIXED)
This version properly freezes the backbone after initialization
"""

import torch
from models.utils.continual_model import ContinualModel
from models.utils.mas_mixin import MASMixin
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace


class LinearMAS(ContinualModel, MASMixin):
    """Linear classifier with MAS regularization and frozen backbone"""
    NAME = 'frozen_backbone_linear_mas_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone: torch.nn.Module, loss: torch.nn.Module,
                 args: Namespace, transform: torch.nn.Module, dataset: ContinualDataset):
        
        # Initialize parent class
        super().__init__(backbone, loss, args, transform, dataset)
        
        # CRITICAL: Freeze the backbone after initialization
        self._freeze_backbone()
        
        # Initialize MAS attributes
        self.omega = {}
        self.old_params = {}
        self.task_count = 0
        
        # Set MAS lambda
        self.mas_lambda = args.mas_lambda if hasattr(args, 'mas_lambda') else 1.0
        
        print(f"\n[LinearMAS] MAS regularization enabled")
        print(f"[LinearMAS] Lambda: {self.mas_lambda}")
        
        # Verify backbone is frozen
        self._verify_frozen()
    
    def _freeze_backbone(self):
        """
        Freeze all backbone parameters.
        Only the classifier should be trainable.
        """
        print("[LinearMAS] Freezing backbone...")
        
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.net.named_parameters():
            # Freeze everything EXCEPT the classifier
            if 'classifier' not in name:
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
                trainable_count += param.numel()
        
        print(f"[LinearMAS] Frozen parameters: {frozen_count:,}")
        print(f"[LinearMAS] Trainable parameters: {trainable_count:,}")
    
    def _verify_frozen(self):
        """
        Verify that backbone is actually frozen.
        Print warning if something is wrong.
        """
        total = 0
        trainable = 0
        
        for param in self.net.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        
        expected_trainable = 5130  # Linear classifier: 512*10 + 10 = 5130
        
        if trainable > expected_trainable * 1.1:  # Allow 10% tolerance
            print(f"[LinearMAS] WARNING: Too many trainable parameters!")
            print(f"[LinearMAS] WARNING: Expected ~{expected_trainable:,}, got {trainable:,}")
            print(f"[LinearMAS] WARNING: Backbone might not be properly frozen!")
        else:
            print(f"[LinearMAS] ✓ Backbone properly frozen")
            print(f"[LinearMAS] ✓ Only classifier is trainable ({trainable:,} params)")
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """Training step with MAS penalty"""
        self.opt.zero_grad()
        
        outputs = self.net(inputs)
        ce_loss = self.loss(outputs, labels)
        mas_loss = self.mas_penalty()
        total_loss = ce_loss + self.mas_lambda * mas_loss
        
        total_loss.backward()
        self.opt.step()
        
        return total_loss.item()
    
    def end_task(self, dataset):
        """Called at the end of each task"""
        MASMixin.end_task(self, dataset)
    
    @staticmethod
    def get_parser(parser):
        parser.add_argument('--mas_lambda', type=float, default=1.0,
                          help='MAS regularization strength (default: 1.0)')
        return parser