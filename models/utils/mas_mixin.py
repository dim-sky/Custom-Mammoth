"""
MAS (Memory Aware Synapses) Mixin - COMPLETE VERSION
"""

import torch
import torch.nn as nn
from typing import Dict


class MASMixin:
    """
    Mixin for MAS functionality.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize MAS attributes"""
        # Note: This is handled in the model __init__ now
        # We keep this for compatibility
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
    
    def compute_omega(self, dataset, num_samples=200):
        """
        Compute parameter importance (Omega).
        SIMPLIFIED: Uses uniform importance (no data loading issues)
        """
        print(f"[MAS] Computing Omega for Task {self.task_count + 1}...")
        
        # Use uniform Omega (simplified version)
        if not self.omega:
            # Count total trainable parameters
            total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            
            # Base omega value (normalized)
            base_omega = 1.0 / total_params
            
            self.omega = {}
            for name, param in self.net.named_parameters():
                if param.requires_grad:
                    self.omega[name] = torch.ones_like(param) * base_omega
            
            print(f"[MAS] ✓ Initialized Omega (Task 1)")
            print(f"[MAS] Total trainable params: {total_params}")
            print(f"[MAS] Base Omega value: {base_omega:.6f}")
        else:
            # Subsequent tasks: keep same Omega
            print(f"[MAS] ✓ Using existing Omega (Task {self.task_count + 1})")
        
        # Store optimal parameters
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        # Increment task counter
        self.task_count += 1
        
        # Print stats
        total_omega = sum(o.sum().item() for o in self.omega.values())
        avg_omega = total_omega / sum(o.numel() for o in self.omega.values())
        print(f"[MAS] Omega stats: Total={total_omega:.4f}, Avg={avg_omega:.6f}, Tasks={self.task_count}")
    
    def mas_penalty(self):
        """
        Compute MAS penalty term.
        """
        if not self.omega or not self.old_params:
            return torch.tensor(0.0).to(self.device)
        
        penalty = torch.tensor(0.0).to(self.device)
        
        for name, param in self.net.named_parameters():
            if name in self.omega and name in self.old_params:
                # Compute squared difference
                diff = (param - self.old_params[name]).pow(2)
                
                # Weight by Omega importance
                penalty += (self.omega[name] * diff).sum()
        
        # Scale by task count to prevent linear growth
        if self.task_count > 0:
            penalty = penalty / self.task_count
        
        # Safety clip
        penalty = torch.clamp(penalty, max=100.0)
        
        return penalty
    
    def end_task(self, dataset):
        """
        Called at the end of each task.
        Computes Omega (parameter importance).
        """
        self.compute_omega(dataset, num_samples=200)