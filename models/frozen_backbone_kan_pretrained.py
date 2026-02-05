"""
KAN Classifier - Frozen Backbone (FIXED)
Baseline model with proper backbone freezing
"""

import torch
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace


class FrozenBackboneKANPretrained(ContinualModel):
    """KAN classifier with frozen pretrained backbone"""
    NAME = 'frozen_backbone_kan_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone: torch.nn.Module, loss: torch.nn.Module,
                 args: Namespace, transform: torch.nn.Module, dataset: ContinualDataset):
        
        super().__init__(backbone, loss, args, transform, dataset)
        
        # CRITICAL FIX: Freeze the backbone
        self._freeze_backbone()
        self._verify_frozen()
        self._print_trainable_summary()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
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
        """Verify backbone is properly frozen"""
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        frozen = total - trainable
        
        # KAN varies by num_grids, roughly 80K-90K for typical config
        expected_trainable = 85000
        
        print("\n" + "="*70)
        print("[VERIFY] Checking freeze status...")
        print("="*70)
        print(f"[VERIFY] Total parameters: {total:,}")
        print(f"[VERIFY] Trainable parameters: {trainable:,}")
        print(f"[VERIFY] Frozen parameters: {frozen:,}")
        print(f"[VERIFY] Expected trainable: ~{expected_trainable:,}")
        
        if trainable > 200000:  # KAN should be < 200K
            print(f"[VERIFY] ❌ ERROR: Too many trainable parameters!")
            print(f"[VERIFY] ❌ Backbone is NOT properly frozen!")
        else:
            print(f"[VERIFY] ✓ SUCCESS: Backbone is properly frozen")
        print("="*70 + "\n")
    
    def _print_trainable_summary(self):
        """Print summary of trainable layers"""
        print("\n" + "="*70)
        print("[SUMMARY] Trainable Layers:")
        print("="*70)
        
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                print(f"[SUMMARY]   ✓ {name}: {param.numel():,} params")
        
        print("="*70 + "\n")
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """Training step"""
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()
        return loss.item()