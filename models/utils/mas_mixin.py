"""
MAS (Memory Aware Synapses) Mixin - DIAGNOSTIC VERSION
This version has extensive debugging to figure out what's wrong
"""

import torch
import torch.nn as nn
from typing import Dict


class MASMixin:
    """
    Mixin class for MAS functionality with detailed debugging
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize MAS attributes"""
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
    
    def compute_omega(self, dataset, num_samples=200):
        """
        Compute parameter importance (Omega).
        DIAGNOSTIC VERSION: Prints detailed information about what's happening
        """
        print(f"\n{'='*70}")
        print(f"[MAS] Computing Omega for Task {self.task_count + 1}...")
        print(f"{'='*70}")
        
        # Debug: Check what parameters exist and their status
        print("\n[MAS DEBUG] Analyzing network parameters:")
        total_params_all = 0
        total_params_trainable = 0
        trainable_by_layer = {}
        
        for name, param in self.net.named_parameters():
            total_params_all += param.numel()
            if param.requires_grad:
                total_params_trainable += param.numel()
                trainable_by_layer[name] = param.numel()
        
        print(f"[MAS DEBUG] Total parameters in network: {total_params_all:,}")
        print(f"[MAS DEBUG] Trainable parameters: {total_params_trainable:,}")
        print(f"[MAS DEBUG] Frozen parameters: {total_params_all - total_params_trainable:,}")
        
        if trainable_by_layer:
            print(f"\n[MAS DEBUG] Trainable layers:")
            for name, count in trainable_by_layer.items():
                print(f"[MAS DEBUG]   {name}: {count:,} params")
        
        # Only initialize Omega once
        if not self.omega:
            # Use ONLY trainable parameters
            trainable_params = [p for p in self.net.parameters() if p.requires_grad]
            total_params = sum(p.numel() for p in trainable_params)
            
            print(f"\n[MAS] Initializing Omega...")
            print(f"[MAS] Using {total_params:,} trainable parameters")
            
            # Safety check
            if total_params == 0:
                print("[MAS ERROR] No trainable parameters found!")
                print("[MAS ERROR] Cannot compute Omega. Aborting.")
                return
            
            # Calculate base importance
            base_omega = 1.0 / total_params
            
            print(f"[MAS] Base Omega value: {base_omega:.10f}")
            print(f"[MAS] (This should be around 0.0002 for Linear classifier)")
            
            # Create Omega dictionary
            self.omega = {}
            omega_count = 0
            for name, param in self.net.named_parameters():
                if param.requires_grad:
                    self.omega[name] = torch.ones_like(param) * base_omega
                    omega_count += param.numel()
            
            print(f"[MAS] Created Omega for {omega_count:,} parameters")
            print(f"[MAS] ✓ Initialized Omega (Task 1)")
        else:
            print(f"[MAS] ✓ Using existing Omega (Task {self.task_count + 1})")
        
        # Store old parameters
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        # Increment task counter
        self.task_count += 1
        
        # Calculate and print statistics
        if self.omega:
            total_omega = sum(o.sum().item() for o in self.omega.values())
            total_elements = sum(o.numel() for o in self.omega.values())
            avg_omega = total_omega / total_elements if total_elements > 0 else 0
            
            print(f"\n[MAS] Omega Statistics:")
            print(f"[MAS]   Total Omega sum: {total_omega:.6f}")
            print(f"[MAS]   Number of elements: {total_elements:,}")
            print(f"[MAS]   Average Omega: {avg_omega:.10f}")
            print(f"[MAS]   Expected average: {1.0/total_elements:.10f}")
            print(f"[MAS]   Tasks completed: {self.task_count}")
            
            # Sanity check
            if abs(avg_omega - 1.0/total_elements) > 1e-8:
                print(f"[MAS WARNING] Average Omega doesn't match expected!")
                print(f"[MAS WARNING] Something might be wrong with initialization.")
        
        print(f"{'='*70}\n")
    
    def mas_penalty(self):
        """
        Calculate MAS penalty with debugging
        """
        if not self.omega or not self.old_params:
            return torch.tensor(0.0).to(self.device)
        
        penalty = torch.tensor(0.0).to(self.device)
        
        for name, param in self.net.named_parameters():
            if name in self.omega and name in self.old_params:
                diff = (param - self.old_params[name]).pow(2)
                penalty += (self.omega[name] * diff).sum()
        
        # Scale by task count
        if self.task_count > 0:
            penalty = penalty / self.task_count
        
        # Safety clip
        penalty = torch.clamp(penalty, max=100.0)
        
        return penalty
    
    def end_task(self, dataset):
        """
        Called at the end of each task
        """
        self.compute_omega(dataset, num_samples=200)