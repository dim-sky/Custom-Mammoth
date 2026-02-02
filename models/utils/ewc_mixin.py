"""
EWC (Elastic Weight Consolidation) Mixin
Reusable component για όλους τους classifiers
"""

import torch
import torch.nn as nn
from typing import Dict


class EWCMixin:
    """
    Mixin class που προσθέτει EWC functionality σε οποιονδήποτε classifier.
    
    Usage:
        class MyModel(EWCMixin, BaseModel):
            ...
    """
    
    def __init__(self):
        """Initialize EWC-specific attributes"""
        self.fisher: Dict[str, torch.Tensor] = {}  # Fisher information
        self.old_params: Dict[str, torch.Tensor] = {}  # Optimal params από previous task
        self.ewc_lambda = 0.0  # EWC penalty strength (set via args)
    
    def compute_fisher(self, dataset, num_samples=1000):
        """
        Υπολογίζει Fisher Information Matrix για τα current weights.
        
        Fisher[i] = E[(∂log p(y|x) / ∂θ_i)²]
        
        Ουσιαστικά: Πόσο "σημαντικό" είναι κάθε weight για το current task.
        
        Args:
            dataset: Training dataset για το current task
            num_samples: Πόσα samples να χρησιμοποιήσουμε
        """
        print(f"[EWC] Computing Fisher Information Matrix...")
        
        # Initialize Fisher dict
        self.fisher = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:  # Μόνο trainable params (classifier)
                self.fisher[name] = torch.zeros_like(param)
        
        # Set model to eval (no dropout, etc.)
        self.net.eval()
        
        # Sample data
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True
        )
        
        samples_seen = 0
        for inputs, labels, _ in loader:
            if samples_seen >= num_samples:
                break
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.net(inputs)
            
            # Compute gradients of log likelihood
            loss = nn.functional.cross_entropy(outputs, labels)
            
            self.opt.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher approximation)
            for name, param in self.net.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.pow(2) * inputs.size(0)
            
            samples_seen += inputs.size(0)
        
        # Normalize by number of samples
        for name in self.fisher:
            self.fisher[name] /= samples_seen
        
        # Store optimal parameters
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        print(f"[EWC] ✓ Fisher computed on {samples_seen} samples")
        
        # Back to train mode
        self.net.train()
    
    def ewc_penalty(self):
        """
        Υπολογίζει το EWC penalty term.
        
        Penalty = Σ_i Fisher_i × (θ_i - θ*_i)²
        
        Returns:
            penalty: Scalar tensor
        """
        if not self.fisher:
            return torch.tensor(0.0).to(self.device)
        
        penalty = torch.tensor(0.0).to(self.device)
        
        for name, param in self.net.named_parameters():
            if name in self.fisher:
                # (current_param - old_param)²
                diff = (param - self.old_params[name]).pow(2)
                
                # Weighted by Fisher importance
                penalty += (self.fisher[name] * diff).sum()
        
        return penalty
    
    def end_task(self, dataset):
        """
        Καλείται στο τέλος κάθε task για να ενημερώσει το Fisher.
        
        Args:
            dataset: Current task dataset
        """
        # Compute Fisher for current task
        self.compute_fisher(dataset, num_samples=1000)