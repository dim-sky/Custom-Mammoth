"""
EWC Mixin - FIXED VERSION (Stable Training)
"""

import torch
import torch.nn as nn
from typing import Dict


class EWCMixin:
    """
    Mixin class που προσθέτει EWC functionality.
    FIXED: Prevents NaN/inf loss με Fisher clipping & normalization.
    """
    
    def __init__(self):
        """Initialize EWC-specific attributes"""
        self.fisher: Dict[str, torch.Tensor] = {}
        self.old_params: Dict[str, torch.Tensor] = {}
        self.ewc_lambda = 0.0
    
    def compute_fisher(self, dataset, num_samples=1000):
        """
        Υπολογίζει Fisher Information Matrix.
        FIXED: Adds stability measures (clipping, normalization).
        """
        print(f"[EWC] Computing Fisher Information Matrix...")
        
        # Initialize Fisher dict
        self.fisher = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param)
        
        # Set model to eval
        self.net.eval()
        
        # Get current task data
        train_dataset = dataset.get_data_loaders()[0].dataset
        
        # Create loader
        from torch.utils.data import DataLoader, Subset
        import numpy as np
        
        if hasattr(train_dataset, '__len__'):
            total_size = len(train_dataset)
            if total_size > num_samples:
                indices = np.random.choice(total_size, num_samples, replace=False)
                sampled_dataset = Subset(train_dataset, indices)
            else:
                sampled_dataset = train_dataset
        else:
            sampled_dataset = train_dataset
        
        loader = DataLoader(
            sampled_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        
        # Accumulate gradients
        samples_seen = 0
        for batch_idx, batch_data in enumerate(loader):
            if samples_seen >= num_samples:
                break
            
            # Unpack batch
            if len(batch_data) == 3:
                inputs, labels, _ = batch_data
            else:
                inputs, labels = batch_data[:2]
            
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
                    self.fisher[name] += param.grad.pow(2) * inputs.size(0)
            
            samples_seen += inputs.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"[EWC] Processed {samples_seen}/{num_samples} samples")
        
        # ========== CRITICAL FIX: Normalize & Clip Fisher ==========
        
        # Normalize by number of samples
        for name in self.fisher:
            self.fisher[name] /= max(samples_seen, 1)
        
        # Clip extreme values (prevent explosion)
        max_fisher = 100.0  # Reasonable upper bound
        for name in self.fisher:
            self.fisher[name] = torch.clamp(self.fisher[name], max=max_fisher)
        
        # Optional: Normalize to [0, 1] range
        for name in self.fisher:
            fisher_max = self.fisher[name].max()
            if fisher_max > 0:
                self.fisher[name] = self.fisher[name] / fisher_max
        
        # ==========================================================
        
        # Store optimal parameters
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        # Debug: Print Fisher statistics
        total_fisher = sum(f.sum().item() for f in self.fisher.values())
        avg_fisher = total_fisher / sum(f.numel() for f in self.fisher.values())
        print(f"[EWC] ✓ Fisher computed on {samples_seen} samples")
        print(f"[EWC] Fisher stats: Total={total_fisher:.2f}, Avg={avg_fisher:.6f}")
        
        # Back to train
        self.net.train()
    
    def ewc_penalty(self):
        """
        Υπολογίζει το EWC penalty term.
        FIXED: Returns reasonable values (no explosion).
        """
        if not self.fisher:
            return torch.tensor(0.0).to(self.device)
        
        penalty = torch.tensor(0.0).to(self.device)
        
        for name, param in self.net.named_parameters():
            if name in self.fisher:
                # Compute difference
                diff = (param - self.old_params[name]).pow(2)
                
                # Weighted by normalized Fisher
                penalty += (self.fisher[name] * diff).sum()
        
        # ========== CRITICAL FIX: Clip penalty ==========
        # Prevent extreme values
        penalty = torch.clamp(penalty, max=1000.0)
        # =================================================
        
        return penalty
    
    def end_task(self, dataset):
        """
        Καλείται στο τέλος κάθε task.
        """
        self.compute_fisher(dataset, num_samples=1000)