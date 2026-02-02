"""
EWC (Elastic Weight Consolidation) Mixin - FINAL FIXED VERSION
Simplified version with correct Fisher magnitude
"""

import torch
import torch.nn as nn
from typing import Dict


class EWCMixin:
    """
    Mixin class for EWC functionality.
    Simplified version: uniform Fisher with correct magnitude
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
        FIXED: Uses Fisher = 1.0 (instead of 0.01) to work with λ=400
        """
        print(f"[EWC] Computing Fisher for Task {self.task_count + 1}...")
        
        # Use uniform Fisher with CORRECT magnitude
        if not self.fisher:
            # First task: initialize
            self.fisher = {}
            for name, param in self.net.named_parameters():
                if param.requires_grad:
                    # CRITICAL FIX: 1.0 instead of 0.01! ⭐⭐⭐
                    # This gives proper penalty strength with λ=400
                    self.fisher[name] = torch.ones_like(param) * 1.0
            
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
        avg_fisher = total_fisher / sum(f.numel() for f in self.fisher.values())
        print(f"[EWC] Fisher stats: Total={total_fisher:.2f}, Avg={avg_fisher:.4f}, Tasks={self.task_count}")
    
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
        
        # Safety clip (higher now that Fisher is larger)
        penalty = torch.clamp(penalty, max=1000.0)
        
        return penalty
    
    def end_task(self, dataset):
        """
        Called at the end of each task.
        
        Args:
            dataset: Current dataset
        """
        self.compute_fisher(dataset, num_samples=200)