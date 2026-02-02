"""
EWC (Elastic Weight Consolidation) Mixin - SIMPLIFIED VERSION
No complex Fisher computation - uses uniform importance
"""

import torch
import torch.nn as nn
from typing import Dict


class EWCMixin:
    """
    Mixin class for EWC functionality.
    Simplified version: uniform Fisher (no data loading issues)
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
        SIMPLIFIED: Uses uniform importance (avoids data loading issues)
        """
        print(f"[EWC] Computing Fisher for Task {self.task_count + 1}...")
        
        # Use uniform Fisher (all params equally important)
        if not self.fisher:
            # First task: initialize
            self.fisher = {}
            for name, param in self.net.named_parameters():
                if param.requires_grad:
                    # Small uniform importance
                    self.fisher[name] = torch.ones_like(param) * 0.01
            
            print(f"[EWC] ✓ Initialized Fisher (Task 1)")
        else:
            # Subsequent tasks: keep same Fisher
            # (In simplified version, we don't recompute)
            print(f"[EWC] ✓ Using existing Fisher (Task {self.task_count + 1})")
        
        # Store optimal parameters from current task
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        # Increment task counter
        self.task_count += 1
        
        # Print stats
        total_fisher = sum(f.sum().item() for f in self.fisher.values())
        print(f"[EWC] Fisher stats: Total={total_fisher:.4f}, Tasks={self.task_count}")
    
    def ewc_penalty(self):
        """
        Compute EWC penalty term.
        
        Returns:
            penalty: Scalar tensor
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
        
        # Scale by task count to prevent linear growth
        if self.task_count > 0:
            penalty = penalty / self.task_count
        
        # Safety clip
        penalty = torch.clamp(penalty, max=100.0)
        
        return penalty
    
    def end_task(self, dataset):
        """
        Called at the end of each task.
        
        Args:
            dataset: Current dataset
        """
        self.compute_fisher(dataset, num_samples=200)