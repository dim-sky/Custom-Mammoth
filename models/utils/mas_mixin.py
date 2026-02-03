"""
MAS (Memory Aware Synapses) Mixin - PROPER IMPLEMENTATION
Computes non-uniform Omega based on output sensitivity (gradients)
Accumulates importance across tasks without task_count division
"""

import torch
import torch.nn as nn
from typing import Dict


class MASMixin:
    """
    Mixin class for MAS functionality with proper importance computation.
    
    Key improvements:
    1. Computes Omega from gradients (output sensitivity) - NOT uniform
    2. Accumulates Omega across tasks (weighted averaging)
    3. No penalty division by task_count (let it grow naturally)
    4. Designed for higher lambda values (10-1000)
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize MAS attributes"""
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
    
    def compute_omega(self, dataset, num_samples=200):
        """
        Compute parameter importance (Omega) from output gradients.
        
        This method measures how much each parameter affects the output.
        Parameters that have large gradients are considered more important.
        
        Algorithm:
        1. Sample data from current task
        2. Forward pass to get outputs
        3. Compute L2 norm of outputs
        4. Backpropagate to get gradients
        5. Omega = absolute value of gradients (importance)
        6. Accumulate across tasks with weighted averaging
        """
        print("\n" + "="*70)
        print(f"[MAS] Computing Omega for Task {self.task_count + 1}...")
        print("="*70)
        
        # Initialize new omega for current task
        new_omega = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                new_omega[name] = torch.zeros_like(param)
        
        # Set model to eval mode (no dropout, batchnorm in eval)
        self.net.eval()
        
        # Get data loader for current task
        try:
            train_loader, _ = dataset.get_data_loaders()
        except Exception as e:
            print(f"[MAS] WARNING: Could not get data loader: {e}")
            print(f"[MAS] Falling back to uniform Omega")
            self._fallback_uniform_omega(new_omega)
            return
        
        # Compute importance from gradients
        print(f"[MAS] Computing importance from {num_samples} samples...")
        samples_processed = 0
        max_batches = max(1, num_samples // dataset.get_batch_size())
        
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            
            # Unpack batch (we only need inputs, not labels)
            if isinstance(batch_data, (list, tuple)):
                inputs = batch_data[0]
                if len(batch_data) > 2:
                    inputs = batch_data[2]  # not_aug_inputs if available
            else:
                inputs = batch_data
            
            inputs = inputs.to(self.device)
            batch_size = inputs.size(0)
            
            # Process each sample individually to get per-sample gradients
            for i in range(batch_size):
                if samples_processed >= num_samples:
                    break
                
                # Zero gradients
                self.net.zero_grad()
                
                # Forward pass for single sample
                single_input = inputs[i:i+1]
                output = self.net(single_input)
                
                # Compute L2 norm of output (measure of output magnitude)
                # This is what we want to preserve - the output pattern
                output_norm = output.pow(2).sum()
                
                # Backward pass - compute how each parameter affects output
                output_norm.backward()
                
                # Accumulate absolute gradients (importance)
                for name, param in self.net.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Use absolute value of gradient as importance measure
                        new_omega[name] += param.grad.abs().detach()
                
                samples_processed += 1
            
            if samples_processed >= num_samples:
                break
            
            # Progress update
            if (batch_idx + 1) % 5 == 0:
                print(f"[MAS] Processed {samples_processed}/{num_samples} samples...")
        
        print(f"[MAS] ✓ Processed {samples_processed} samples")
        
        # Normalize by number of samples
        if samples_processed > 0:
            for name in new_omega:
                new_omega[name] /= samples_processed
        
        # Accumulate omega across tasks (online EWC-style)
        if not self.omega:
            # First task - just use computed omega
            self.omega = new_omega
            print(f"[MAS] ✓ Initialized Omega (Task 1)")
        else:
            # Subsequent tasks - weighted average with previous omega
            # This accumulates importance: important for multiple tasks = very important
            alpha = 1.0 / (self.task_count + 1)  # Weight for new task
            
            print(f"[MAS] Accumulating Omega (alpha={alpha:.4f})...")
            for name in self.omega:
                if name in new_omega:
                    # Weighted average: old importance + new importance
                    self.omega[name] = (1 - alpha) * self.omega[name] + alpha * new_omega[name]
            
            print(f"[MAS] ✓ Accumulated Omega (Task {self.task_count + 1})")
        
        # Normalize omega to prevent explosion
        # Scale so that sum = number of parameters (not 1.0)
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        current_sum = sum(o.sum().item() for o in self.omega.values())
        
        if current_sum > 0:
            scale_factor = total_params / current_sum
            for name in self.omega:
                self.omega[name] *= scale_factor
        
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
    
    def _fallback_uniform_omega(self, new_omega):
        """Fallback to uniform omega if gradient computation fails"""
        print("[MAS] Using uniform Omega as fallback...")
        
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        base_omega = 1.0
        
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                new_omega[name] = torch.ones_like(param) * base_omega
        
        if not self.omega:
            self.omega = new_omega
        else:
            alpha = 1.0 / (self.task_count + 1)
            for name in self.omega:
                self.omega[name] = (1 - alpha) * self.omega[name] + alpha * new_omega[name]
        
        self.old_params = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        self.task_count += 1
        self._print_omega_stats()
    
    def _print_omega_stats(self):
        """Print detailed statistics about Omega values"""
        print("\n" + "="*70)
        print("[MAS] Omega Statistics:")
        print("="*70)
        
        # Overall stats
        total_omega = sum(o.sum().item() for o in self.omega.values())
        total_elements = sum(o.numel() for o in self.omega.values())
        avg_omega = total_omega / total_elements if total_elements > 0 else 0
        
        print(f"[MAS]   Total Omega sum: {total_omega:.2f}")
        print(f"[MAS]   Average Omega: {avg_omega:.6f}")
        print(f"[MAS]   Protected parameters: {total_elements:,}")
        print(f"[MAS]   Tasks accumulated: {self.task_count}")
        
        # Per-layer stats (top 5 most important)
        layer_importance = []
        for name, omega in self.omega.items():
            layer_sum = omega.sum().item()
            layer_avg = omega.mean().item()
            layer_max = omega.max().item()
            layer_importance.append((name, layer_sum, layer_avg, layer_max))
        
        # Sort by total importance
        layer_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n[MAS]   Top 5 Most Important Layers:")
        for i, (name, total, avg, max_val) in enumerate(layer_importance[:5]):
            print(f"[MAS]     {i+1}. {name}")
            print(f"[MAS]        Total: {total:.4f}, Avg: {avg:.6f}, Max: {max_val:.6f}")
        
        print("="*70 + "\n")
    
    def mas_penalty(self):
        """
        Compute MAS penalty term.
        
        CRITICAL CHANGES:
        1. NO division by task_count (let penalty grow with tasks)
        2. Uses accumulated Omega (grows with importance)
        3. Designed for higher lambda values (10-1000)
        
        Formula: penalty = λ × Σ Omega[i] × (current[i] - old[i])²
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
        
        # NO division by task_count - let it grow!
        # This is intentional - accumulated importance × accumulated changes
        
        # Safety clip to prevent numerical overflow
        # Higher max because we expect larger penalties now
        penalty = torch.clamp(penalty, max=10000.0)
        
        return penalty
    
    def end_task(self, dataset):
        """
        Called at the end of each task.
        Computes Omega (parameter importance) from gradients.
        """
        self.compute_omega(dataset, num_samples=200)