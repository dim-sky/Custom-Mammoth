"""
MLP Classifier with MAS - Frozen Backbone
"""

import torch
from models.utils.continual_model import ContinualModel
from models.utils.mas_mixin import MASMixin
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace


class MLPMAS(ContinualModel, MASMixin):
    """MLP classifier with MAS regularization and frozen backbone"""
    NAME = 'frozen_backbone_mlp_mas_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone: torch.nn.Module, loss: torch.nn.Module,
                 args: Namespace, transform: torch.nn.Module, dataset: ContinualDataset):
        
        super().__init__(backbone, loss, args, transform, dataset)
        
        self._freeze_backbone()
        self._verify_frozen()
        self._print_trainable_summary()
        
        self.omega = {}
        self.old_params = {}
        self.task_count = 0
        
        self.mas_lambda = args.mas_lambda if hasattr(args, 'mas_lambda') else 1.0
        
        print(f"\n{'='*70}")
        print(f"[MLPMAS] MAS regularization enabled")
        print(f"[MLPMAS] Lambda: {self.mas_lambda}")
        print(f"{'='*70}\n")
    
    def _freeze_backbone(self):
        print("\n" + "="*70)
        print("[FREEZE] Freezing backbone parameters...")
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
        
        print(f"[FREEZE] ✓ Frozen parameters: {frozen_count:,}")
        print(f"[FREEZE] ✓ Trainable parameters: {trainable_count:,}")
        print("="*70 + "\n")
    
    def _verify_frozen(self):
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        frozen = total - trainable
        
        expected_trainable = 133898
        
        print("\n" + "="*70)
        print("[VERIFY] Checking freeze status...")
        print("="*70)
        print(f"[VERIFY] Total parameters: {total:,}")
        print(f"[VERIFY] Trainable parameters: {trainable:,}")
        print(f"[VERIFY] Frozen parameters: {frozen:,}")
        print(f"[VERIFY] Expected trainable: ~{expected_trainable:,}")
        print(f"[VERIFY] Percentage frozen: {100 * frozen / total:.2f}%")
        
        if trainable > expected_trainable * 1.2:
            print(f"[VERIFY] ❌ ERROR: Too many trainable parameters!")
        else:
            print(f"[VERIFY] ✓ SUCCESS: Backbone is properly frozen")
        print("="*70 + "\n")
    
    def _print_trainable_summary(self):
        print("\n" + "="*70)
        print("[SUMMARY] Trainable Layers:")
        print("="*70)
        
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                print(f"[SUMMARY]   ✓ {name}: {param.numel():,} params")
        
        print("="*70 + "\n")
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        
        outputs = self.net(inputs)
        ce_loss = self.loss(outputs, labels)
        mas_loss = self.mas_penalty()
        total_loss = ce_loss + self.mas_lambda * mas_loss
        
        total_loss.backward()
        self.opt.step()
        
        return total_loss.item()
    
    def end_task(self, dataset):
        MASMixin.end_task(self, dataset)
    
    @staticmethod
    def get_parser(parser):
        parser.add_argument('--mas_lambda', type=float, default=1.0,
                          help='MAS regularization strength (recommended: 0.5-10, default: 1.0)')
        return parser