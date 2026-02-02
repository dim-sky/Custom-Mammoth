"""
EWC Mixin - FIXED VERSION (Online EWC με Proper Accumulation)
"""

import torch
import torch.nn as nn
from typing import Dict


class EWCMixin:
    """
    Mixin class που προσθέτει EWC functionality.
    FIXED: Proper Fisher accumulation (Online EWC)
    """
    
    def __init__(self):
        """Initialize EWC-specific attributes"""
        self.fisher: Dict[str, torch.Tensor] = {}  # Accumulated Fisher
        self.old_params: Dict[str, torch.Tensor] = {}
        self.ewc_lambda = 0.0
        self.task_count = 0  # Track number of tasks
    
    def compute_fisher(self, dataset, num_samples=200):
        """
        Υπολογίζει Fisher για CURRENT task και τον ACCUMULATES.
        
        FIXED: Uses Online EWC - accumulates Fisher across tasks
        """
        print(f"[EWC] Computing Fisher for Task {self.task_count + 1}...")
        
        # ========== STEP 1: Compute NEW Fisher για current task ==========
        new_fisher = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                new_fisher[name] = torch.zeros_like(param)
        
        # Set model to eval
        self.net.eval()
        
        # ========== Get current task data ==========
        try:
            train_loader = dataset.get_data_loaders()[0]
        except (IndexError, ValueError) as e:
            print(f"[EWC] Warning: Could not get data loader: {e}")
            print(f"[EWC] Skipping Fisher computation for this task")
            self.task_count += 1
            return
        
        # Sample data
        samples_seen = 0
        batch_count = 0
        max_batches = max(1, num_samples // 32)  # Limit batches
        
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            
            # Unpack batch
            try:
                if len(batch_data) == 3:
                    inputs, labels, _ = batch_data
                else:
                    inputs, labels = batch_data[:2]
            except (ValueError, TypeError):
                print(f"[EWC] Warning: Could not unpack batch, skipping")
                continue
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.net(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            
            # Backward
            self.opt.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.net.named_parameters():
                if param.requires_grad and param.grad is not None:
                    new_fisher[name] += param.grad.pow(2) * inputs.size(0)
            
            samples_seen += inputs.size(0)
            batch_count += 1
        
        if samples_seen == 0:
            print(f"[EWC] Warning: No samples processed, skipping Fisher")
            self.task_count += 1
            return
        
        # Normalize NEW Fisher
        for name in new_fisher:
            new_fisher[name] /= samples_seen
        
        # ========== CRITICAL FIX: Clip & Normalize NEW Fisher ==========
        max_fisher = 10.0  # Lower threshold (was 100)
        for name in new_fisher:
            # Clip extreme values
            new_fisher[name] = torch.clamp(new_fisher[name], max=max_fisher)
            
            # Normalize to [0, 1]
            fisher_max = new_fisher[name].max()
            if fisher_max > 1e-8:  # Avoid division by near-zero
                new_fisher[name] = new_fisher[name] / fisher_max
        
        # ========== STEP 2: ACCUMULATE Fisher (Online EWC) ==========
        if not self.fisher:  # First task
            # Initialize Fisher
            self.fisher = new_fisher
            print(f"[EWC] ✓ Initialized Fisher (Task 1)")
        else:  # Subsequent tasks
            # AVERAGE Fisher across tasks (not sum!)
            # This prevents explosion
            alpha = 1.0 / (self.task_count + 1)  # Weight for new task
            
            for name in self.fisher:
                # Weighted average: old Fisher * (1-α) + new Fisher * α
                self.fisher[name] = (1 - alpha) * self.fisher[name] + alpha * new_fisher[name]
            
            print(f"[EWC] ✓ Accumulated Fisher (Task {self.task_count + 1})")
        
        # ========== STEP 3: Store optimal parameters ==========
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        # Debug: Print stats
        total_fisher = sum(f.sum().item() for f in self.fisher.values())
        print(f"[EWC] Fisher stats: Total={total_fisher:.4f}, Tasks={self.task_count + 1}")
        
        # Increment task counter
        self.task_count += 1
        
        # Back to train
        self.net.train()
    
    def ewc_penalty(self):
        """
        Υπολογίζει το EWC penalty.
        FIXED: Uses accumulated Fisher
        """
        if not self.fisher:
            return torch.tensor(0.0).to(self.device)
        
        penalty = torch.tensor(0.0).to(self.device)
        
        for name, param in self.net.named_parameters():
            if name in self.fisher:
                diff = (param - self.old_params[name]).pow(2)
                penalty += (self.fisher[name] * diff).sum()
        
        # ========== CRITICAL: Scale penalty by task count ==========
        # Prevents linear growth as tasks increase
        if self.task_count > 0:
            penalty = penalty / self.task_count
        
        # Final safety clip
        penalty = torch.clamp(penalty, max=100.0)
        
        return penalty
    
    def end_task(self, dataset):
        """Called after each task"""
        self.compute_fisher(dataset, num_samples=200)