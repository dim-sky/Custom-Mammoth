"""
EWC (Elastic Weight Consolidation) Mixin - FIXED VERSION
"""

import torch
import torch.nn as nn
from typing import Dict


class EWCMixin:
    """
    Mixin class που προσθέτει EWC functionality.
    """
    
    def __init__(self):
        """Initialize EWC-specific attributes"""
        self.fisher: Dict[str, torch.Tensor] = {}
        self.old_params: Dict[str, torch.Tensor] = {}
        self.ewc_lambda = 0.0
    
    def compute_fisher(self, dataset, num_samples=1000):
        """
        Υπολογίζει Fisher Information Matrix.
        
        FIXED: Works με Mammoth datasets που δεν έχουν __len__
        """
        print(f"[EWC] Computing Fisher Information Matrix...")
        
        # Initialize Fisher dict
        self.fisher = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param)
        
        # Set model to eval
        self.net.eval()
        
        # ========== FIX: Use Mammoth's train_loader ==========
        # Mammoth datasets έχουν custom interface
        # Χρησιμοποιούμε το train_loader από το dataset
        
        # Get current task data
        train_dataset = dataset.get_data_loaders()[0].dataset
        
        # Create loader manually
        from torch.utils.data import DataLoader, Subset
        import numpy as np
        
        # Sample indices (if dataset too large)
        if hasattr(train_dataset, '__len__'):
            total_size = len(train_dataset)
            if total_size > num_samples:
                indices = np.random.choice(total_size, num_samples, replace=False)
                sampled_dataset = Subset(train_dataset, indices)
            else:
                sampled_dataset = train_dataset
        else:
            # Fallback: just use full dataset
            sampled_dataset = train_dataset
        
        loader = DataLoader(
            sampled_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0  # Important για stability
        )
        
        # Accumulate gradients
        samples_seen = 0
        for batch_idx, batch_data in enumerate(loader):
            if samples_seen >= num_samples:
                break
            
            # Unpack batch (Mammoth format)
            if len(batch_data) == 3:
                inputs, labels, _ = batch_data
            else:
                inputs, labels = batch_data[:2]
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.net(inputs)
            
            # Compute loss
            loss = nn.functional.cross_entropy(outputs, labels)
            
            # Backward
            self.opt.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.net.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.pow(2) * inputs.size(0)
            
            samples_seen += inputs.size(0)
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"[EWC] Processed {samples_seen}/{num_samples} samples")
        
        # Normalize
        for name in self.fisher:
            self.fisher[name] /= max(samples_seen, 1)  # Avoid division by 0
        
        # Store optimal parameters
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        print(f"[EWC] ✓ Fisher computed on {samples_seen} samples")
        
        # Back to train
        self.net.train()
    
    def ewc_penalty(self):
        """
        Υπολογίζει το EWC penalty term.
        """
        if not self.fisher:
            return torch.tensor(0.0).to(self.device)
        
        penalty = torch.tensor(0.0).to(self.device)
        
        for name, param in self.net.named_parameters():
            if name in self.fisher:
                diff = (param - self.old_params[name]).pow(2)
                penalty += (self.fisher[name] * diff).sum()
        
        return penalty
    
    def end_task(self, dataset):
        """
        Καλείται στο τέλος κάθε task.
        """
        # Compute Fisher for current task
        self.compute_fisher(dataset, num_samples=1000)