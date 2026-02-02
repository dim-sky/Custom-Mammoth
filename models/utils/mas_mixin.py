"""
MAS (Memory Aware Synapses) Mixin - CORRECTED VERSION

This version properly handles frozen backbones by counting
ONLY trainable parameters, not the entire network.
"""

import torch
import torch.nn as nn
from typing import Dict


class MASMixin:
    """
    Mixin class that adds MAS (Memory Aware Synapses) functionality
    to continual learning models.
    
    MAS works by:
    1. Measuring parameter importance (Omega) after each task
    2. Adding a penalty when parameters change too much in future tasks
    3. This prevents catastrophic forgetting
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize MAS attributes.
        
        Note: This mixin is designed to work with multiple inheritance.
        We accept *args and **kwargs to pass them to other parent classes.
        """
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
    
    def compute_omega(self, dataset, num_samples=200):
        """
        Compute parameter importance (Omega).
        
        This method calculates how "important" each parameter is.
        In this simplified version, we use uniform importance
        (all parameters equally important).
        
        CRITICAL FIX: We only count TRAINABLE parameters,
        not the frozen backbone. This prevents the Omega values
        from being too small.
        """
        print(f"[MAS] Computing Omega for Task {self.task_count + 1}...")
        
        # Only initialize Omega once (first task)
        if not self.omega:
            # Count ONLY parameters that can actually change (trainable)
            trainable_params = [p for p in self.net.parameters() if p.requires_grad]
            total_params = sum(p.numel() for p in trainable_params)
            
            # Calculate base importance value
            # We normalize by total params so the sum equals 1.0
            base_omega = 1.0 / total_params
            
            # Create Omega dictionary
            # Each trainable parameter gets the same base importance
            self.omega = {}
            for name, param in self.net.named_parameters():
                if param.requires_grad:
                    # All parameters get uniform importance
                    self.omega[name] = torch.ones_like(param) * base_omega
            
            print(f"[MAS] ✓ Initialized Omega (Task 1)")
            print(f"[MAS] Trainable parameters: {total_params:,}")
            print(f"[MAS] Base Omega value per parameter: {base_omega:.8f}")
        else:
            # For subsequent tasks, keep using the same Omega
            # (In this simplified version, we don't recompute)
            print(f"[MAS] ✓ Using existing Omega (Task {self.task_count + 1})")
        
        # Save the current parameter values
        # These will be the "target" values that we try to stay close to
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        # Increment task counter
        self.task_count += 1
        
        # Print statistics for debugging
        total_omega = sum(o.sum().item() for o in self.omega.values())
        avg_omega = total_omega / sum(o.numel() for o in self.omega.values())
        print(f"[MAS] Omega statistics:")
        print(f"[MAS]   Total Omega: {total_omega:.4f}")
        print(f"[MAS]   Average Omega: {avg_omega:.8f}")
        print(f"[MAS]   Tasks completed: {self.task_count}")
    
    def mas_penalty(self):
        """
        Calculate the MAS penalty (regularization term).
        
        This penalty measures how much the parameters have changed
        from their "optimal" values after the previous task.
        
        Formula: penalty = Σ Omega[i] × (current_param[i] - old_param[i])²
        
        The penalty is:
        - Zero if parameters haven't changed
        - Large if important parameters have changed a lot
        """
        # If we haven't computed Omega yet, no penalty
        if not self.omega or not self.old_params:
            return torch.tensor(0.0).to(self.device)
        
        penalty = torch.tensor(0.0).to(self.device)
        
        # For each trainable parameter
        for name, param in self.net.named_parameters():
            if name in self.omega and name in self.old_params:
                # Calculate how much it changed (squared difference)
                diff = (param - self.old_params[name]).pow(2)
                
                # Weight the change by importance (Omega)
                # Important parameters get penalized more for changing
                penalty += (self.omega[name] * diff).sum()
        
        # Divide by number of tasks to prevent linear growth
        # This keeps the penalty magnitude consistent as we learn more tasks
        if self.task_count > 0:
            penalty = penalty / self.task_count
        
        # Safety: clip penalty to prevent extreme values
        # This prevents numerical instability
        penalty = torch.clamp(penalty, max=100.0)
        
        return penalty
    
    def end_task(self, dataset):
        """
        Called at the end of each task.
        
        This method computes the Omega (importance) values
        based on the current task, then saves the current
        parameter values as the "optimal" values to protect.
        """
        self.compute_omega(dataset, num_samples=200)