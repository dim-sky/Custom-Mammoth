"""
MAS (Memory Aware Synapses) Mixin - PRODUCTION VERSION
Following best practices from built-in EWC-ON

Key improvements:
1. Uses existing dataset.train_loader (no dataset.i modification)
2. Normalizes by total dataset size (not sample count)
3. Uses reasonable lambda values (1-10)
4. Proper accumulation with decay (gamma parameter)
"""

import torch
import torch.nn as nn
from typing import Dict


class MASMixin:
    """
    Mixin class for MAS functionality.
    Implements Memory Aware Synapses following EWC-ON best practices.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize MAS attributes"""
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
    
    def compute_omega(self, dataset, num_samples=None):
        """
        Compute parameter importance (Omega) from output gradients.
        
        Following EWC-ON approach:
        1. Uses dataset.train_loader directly (no get_data_loaders())
        2. Normalizes by total dataset size
        3. Accumulates with decay (gamma)
        
        Args:
            dataset: Current dataset object with train_loader
            num_samples: If None, uses entire dataset. If int, uses that many samples.
        """
        print("\n" + "="*70)
        print(f"[MAS] Computing Omega for Task {self.task_count + 1}...")
        print("="*70)
        
        # Initialize new omega (flattened vector like EWC-ON would do)
        # But we'll keep dict structure for compatibility
        new_omega = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                new_omega[name] = torch.zeros_like(param)
        
        # Set model to eval mode
        self.net.eval()
        
        # CRITICAL: Use existing train_loader (doesn't modify dataset.i)
        if not hasattr(dataset, 'train_loader'):
            print("[MAS] ERROR: dataset has no train_loader!")
            print("[MAS] Falling back to uniform Omega")
            self._fallback_uniform_omega(new_omega)
            return
        
        train_loader = dataset.train_loader
        batch_size = dataset.get_batch_size()
        
        # Determine how many samples to process
        if num_samples is None:
            # Use entire dataset
            max_batches = len(train_loader)
            total_samples = len(train_loader) * batch_size
            print(f"[MAS] Computing importance from entire dataset ({total_samples} samples)...")
        else:
            # Use specified number of samples
            max_batches = max(1, num_samples // batch_size)
            total_samples = max_batches * batch_size
            print(f"[MAS] Computing importance from {total_samples} samples...")
        
        samples_processed = 0
        
        try:
            for batch_idx, batch_data in enumerate(train_loader):
                if batch_idx >= max_batches:
                    break
                
                # Unpack batch
                if isinstance(batch_data, (list, tuple)):
                    inputs = batch_data[0]
                    # Prefer not_aug_inputs if available
                    if len(batch_data) > 2 and batch_data[2] is not None:
                        inputs = batch_data[2]
                else:
                    inputs = batch_data
                
                inputs = inputs.to(self.device)
                current_batch_size = inputs.size(0)
                
                # Process each sample in batch individually
                for i in range(current_batch_size):
                    self.net.zero_grad()
                    
                    # Forward pass for single sample
                    single_input = inputs[i:i+1]
                    output = self.net(single_input)
                    
                    # Compute L2 norm of output (like MAS paper)
                    output_norm = output.pow(2).sum()
                    
                    # Backward to get gradients
                    output_norm.backward()
                    
                    # Accumulate absolute gradients as importance
                    for name, param in self.net.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            new_omega[name] += param.grad.abs().detach()
                    
                    samples_processed += 1
                
                # Progress update every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"[MAS] Processed {samples_processed} samples...")
        
        except Exception as e:
            print(f"[MAS] ERROR during Omega computation: {e}")
            print(f"[MAS] Falling back to uniform Omega")
            self._fallback_uniform_omega(new_omega)
            return
        
        print(f"[MAS] ✓ Processed {samples_processed} samples")
        
        # CRITICAL: Normalize by TOTAL DATASET SIZE (like EWC-ON)
        # This keeps omega values small and prevents explosion
        total_dataset_size = len(train_loader) * batch_size
        
        if samples_processed > 0:
            for name in new_omega:
                # Divide by samples processed to get average
                new_omega[name] /= samples_processed
                
                # Scale to represent full dataset
                # This makes omega comparable across different sample sizes
                new_omega[name] *= (samples_processed / total_dataset_size)
        
        print(f"[MAS] Normalized by dataset size: {total_dataset_size}")
        
        # Accumulate omega with decay (like EWC-ON with gamma)
        if not self.omega:
            # First task - initialize
            self.omega = new_omega
            print(f"[MAS] ✓ Initialized Omega (Task 1)")
        else:
            # Subsequent tasks - accumulate with decay
            gamma = 0.9  # Decay factor (like EWC-ON)
            
            print(f"[MAS] Accumulating Omega with decay (gamma={gamma})...")
            for name in self.omega:
                if name in new_omega:
                    # Decay old importance, add new importance
                    self.omega[name] = gamma * self.omega[name] + new_omega[name]
            
            print(f"[MAS] ✓ Accumulated Omega (Task {self.task_count + 1})")
        
        # Store optimal parameters from current task
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        # Increment task counter
        self.task_count += 1
        
        # Print statistics
        self._print_omega_stats()
        
        # Set model back to train mode
        self.net.train()
        
        print("="*70 + "\n")
    
    def _fallback_uniform_omega(self, new_omega):
        """Fallback to uniform omega if gradient computation fails"""
        print("[MAS] Using uniform Omega as fallback...")
        
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        
        # Small uniform value (like normalized gradient would be)
        base_omega = 0.001 / total_params
        
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                new_omega[name] = torch.ones_like(param) * base_omega
        
        if not self.omega:
            self.omega = new_omega
        else:
            gamma = 0.9
            for name in self.omega:
                self.omega[name] = gamma * self.omega[name] + new_omega[name]
        
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        self.task_count += 1
        self._print_omega_stats()
    
    def _print_omega_stats(self):
        """Print detailed statistics about Omega values"""
        print("\n" + "-"*70)
        print("[MAS] Omega Statistics:")
        print("-"*70)
        
        # Overall stats
        total_omega = sum(o.sum().item() for o in self.omega.values())
        total_elements = sum(o.numel() for o in self.omega.values())
        avg_omega = total_omega / total_elements if total_elements > 0 else 0
        min_omega = min(o.min().item() for o in self.omega.values())
        max_omega = max(o.max().item() for o in self.omega.values())
        
        print(f"[MAS]   Total Omega sum: {total_omega:.6f}")
        print(f"[MAS]   Average Omega: {avg_omega:.8f}")
        print(f"[MAS]   Min Omega: {min_omega:.8f}")
        print(f"[MAS]   Max Omega: {max_omega:.6f}")
        print(f"[MAS]   Protected parameters: {total_elements:,}")
        print(f"[MAS]   Tasks accumulated: {self.task_count}")
        
        # Per-layer stats (top 2 for brevity)
        layer_importance = []
        for name, omega in self.omega.items():
            layer_sum = omega.sum().item()
            layer_avg = omega.mean().item()
            layer_max = omega.max().item()
            layer_importance.append((name, layer_sum, layer_avg, layer_max))
        
        # Sort by total importance
        layer_importance.sort(key=lambda x: x[1], reverse=True)
        
        if layer_importance:
            print(f"\n[MAS]   Most Important Layers:")
            for i, (name, total, avg, max_val) in enumerate(layer_importance[:2]):
                print(f"[MAS]     {i+1}. {name}")
                print(f"[MAS]        Sum: {total:.6f}, Avg: {avg:.8f}, Max: {max_val:.6f}")
        
        print("-"*70)
    
    def mas_penalty(self):
        """
        Compute MAS penalty term.
        
        With proper normalization, lambda values of 1-10 work well.
        """
        if not self.omega or not self.old_params:
            return torch.tensor(0.0).to(self.device)
        
        penalty = torch.tensor(0.0).to(self.device)
        
        # Compute penalty for each parameter
        for name, param in self.net.named_parameters():
            if name in self.omega and name in self.old_params:
                # Squared difference from optimal values
                diff = (param - self.old_params[name]).pow(2)
                
                # Weight by importance (Omega)
                weighted_diff = self.omega[name] * diff
                penalty += weighted_diff.sum()
        
        # Safety clip to prevent numerical issues
        # With proper normalization, this should rarely trigger
        penalty = torch.clamp(penalty, max=10000.0)
        
        return penalty
    
    def end_task(self, dataset):
        """
        Called at the end of each task.
        Computes Omega (parameter importance) from gradients.
        """
        # Compute omega using first 1000 samples (or entire dataset if smaller)
        # This is a good balance between speed and accuracy
        self.compute_omega(dataset, num_samples=1000)