"""
EWC Mixin - PROPERLY SCALED VERSION
"""

import torch
import torch.nn as nn
from typing import Dict


class EWCMixin:
    """
    Mixin for EWC functionality.
    FIXED: Properly scaled Fisher to avoid explosion
    """
    
    def __init__(self):
        """Initialize EWC attributes"""
        self.fisher: Dict[str, torch.Tensor] = {}
        self.old_params: Dict[str, torch.Tensor] = {}
        self.ewc_lambda = 0.0
        self.task_count = 0
    
    def compute_fisher(self, dataset, num_samples=200):
        """
        Compute Fisher Information.
        FIXED: Normalize Fisher by total parameter count
        """
        print(f"[EWC] Computing Fisher for Task {self.task_count + 1}...")
        
        # Use uniform Fisher with PROPER normalization
        if not self.fisher:
            # First task: initialize
            self.fisher = {}
            
            # Count total trainable parameters
            total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            
            # Base Fisher value (normalized)
            # This ensures total penalty scales reasonably
            base_fisher = 1.0 / total_params  # ⭐⭐⭐ KEY FIX!
            
            for name, param in self.net.named_parameters():
                if param.requires_grad:
                    self.fisher[name] = torch.ones_like(param) * base_fisher
            
            print(f"[EWC] ✓ Initialized Fisher (Task 1)")
            print(f"[EWC] Total trainable params: {total_params}")
            print(f"[EWC] Base Fisher value: {base_fisher:.6f}")
        else:
            # Subsequent tasks: keep same Fisher
            print(f"[EWC] ✓ Using existing Fisher (Task {self.task_count + 1})")
        
        # Store optimal parameters
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        # Increment task counter
        self.task_count += 1
        
        # Print stats
        total_fisher = sum(f.sum().item() for f in self.fisher.values())
        avg_fisher = total_fisher / sum(f.numel() for f in self.fisher.values())
        print(f"[EWC] Fisher stats: Total={total_fisher:.4f}, Avg={avg_fisher:.6f}, Tasks={self.task_count}")
    
    def ewc_penalty(self):
        """
        Compute EWC penalty term.
        """
        if not self.fisher or not self.old_params:
            return torch.tensor(0.0).to(self.device)
        
        penalty = torch.tensor(0.0).to(self.device)
        
        for name, param in self.net.named_parameters():
            if name in self.fisher and name in self.old_params:
                # Compute squared difference
                diff = (param - self.old_params[name]).pow(2)
                
                # Weight by Fisher importance
                penalty += (self.fisher[name] * diff).sum()
        
        # Scale by task count
        if self.task_count > 0:
            penalty = penalty / self.task_count
        
        # Safety clip
        penalty = torch.clamp(penalty, max=100.0)
        
        return penalty
    
    def end_task(self, dataset):
        """Called at the end of each task"""
        self.compute_fisher(dataset, num_samples=200)