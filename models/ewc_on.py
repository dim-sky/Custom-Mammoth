"""
EWC Mixin - ULTRA-STABLE VERSION
"""

import torch
import torch.nn as nn
from typing import Dict


class EWCMixin:
    """Ultra-stable EWC with aggressive normalization"""
    
    def __init__(self):
        self.fisher: Dict[str, torch.Tensor] = {}
        self.old_params: Dict[str, torch.Tensor] = {}
        self.ewc_lambda = 0.0
        self.task_count = 0
    
    def compute_fisher(self, dataset, num_samples=200):
        """Compute normalized Fisher"""
        print(f"[EWC] Computing Fisher for Task {self.task_count + 1}...")
        
        # Initialize
        new_fisher = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                new_fisher[name] = torch.zeros_like(param)
        
        self.net.eval()
        
        # Get loader safely
        try:
            train_loader = dataset.train_loader
            # Test if empty
            test_iter = iter(train_loader)
            _ = next(test_iter)
        except (AttributeError, StopIteration, TypeError) as e:
            print(f"[EWC] Cannot access train_loader: {e}")
            print(f"[EWC] Skipping Fisher for this task")
            self.task_count += 1
            return
        
        # Compute Fisher
        samples_seen = 0
        for batch_idx, batch_data in enumerate(train_loader):
            if samples_seen >= num_samples:
                break
            
            try:
                inputs, labels = batch_data[0], batch_data[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.net(inputs)
                loss = nn.functional.cross_entropy(outputs, labels)
                
                self.opt.zero_grad()
                loss.backward()
                
                for name, param in self.net.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        new_fisher[name] += param.grad.pow(2)
                
                samples_seen += inputs.size(0)
            except Exception as e:
                print(f"[EWC] Batch error: {e}")
                continue
        
        if samples_seen == 0:
            print(f"[EWC] No samples processed!")
            self.task_count += 1
            return
        
        # ULTRA-AGGRESSIVE NORMALIZATION
        for name in new_fisher:
            # Normalize by samples
            new_fisher[name] /= samples_seen
            # Clip
            new_fisher[name] = torch.clamp(new_fisher[name], max=0.1)
            # Per-param normalize
            fmax = new_fisher[name].max()
            if fmax > 1e-8:
                new_fisher[name] /= fmax
        
        # GLOBAL normalize to sum=1
        total = sum(f.sum().item() for f in new_fisher.values())
        if total > 0:
            for name in new_fisher:
                new_fisher[name] /= total
        
        # Accumulate
        if not self.fisher:
            self.fisher = new_fisher
        else:
            for name in self.fisher:
                self.fisher[name] = 0.5 * self.fisher[name] + 0.5 * new_fisher[name]
        
        # Checkpoint
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        total = sum(f.sum().item() for f in self.fisher.values())
        print(f"[EWC] ✓ Fisher computed. Total={total:.6f}")
        
        self.task_count += 1
        self.net.train()
    
    def ewc_penalty(self):
        """Compute penalty"""
        if not self.fisher:
            return torch.tensor(0.0).to(self.device)
        
        penalty = torch.tensor(0.0).to(self.device)
        for name, param in self.net.named_parameters():
            if name in self.fisher:
                diff = (param - self.old_params[name]).pow(2)
                penalty += (self.fisher[name] * diff).sum()
        
        # Safety clip
        penalty = torch.clamp(penalty, max=100.0)
        return penalty
    
    def end_task(self, dataset):
        self.compute_fisher(dataset, num_samples=200)