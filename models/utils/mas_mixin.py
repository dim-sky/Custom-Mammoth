"""
MAS (Memory Aware Synapses) Mixin - PROPERLY SCALED VERSION
Computes non-uniform Omega with CORRECT scaling for small classifiers
"""

import torch
import torch.nn as nn
from typing import Dict


class MASMixin:
    """
    Mixin class for MAS functionality with proper scaling.
    
    KEY FIX: Omega is normalized to sum=1.0, not sum=num_params
    This prevents penalty explosion with high lambda values.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize MAS attributes"""
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
    
    def compute_omega(self, dataset, num_samples=200):
        """
        Compute parameter importance (Omega) from output gradients.
        CRITICAL FIX: Normalize Omega to sum=1.0 to prevent explosion
        """
        print("\n" + "="*70)
        print(f"[MAS] Computing Omega for Task {self.task_count + 1}...")
        print("="*70)
        
        # Initialize new omega for current task
        new_omega = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                new_omega[name] = torch.zeros_like(param)
        
        # Set model to eval mode
        self.net.eval()
        
        # Get data loader
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
            
            # Unpack batch
            if isinstance(batch_data, (list, tuple)):
                inputs = batch_data[0]
                if len(batch_data) > 2:
                    inputs = batch_data[2]
            else:
                inputs = batch_data
            
            inputs = inputs.to(self.device)
            batch_size = inputs.size(0)
            
            # Process samples
            for i in range(min(batch_size, num_samples - samples_processed)):
                self.net.zero_grad()
                
                single_input = inputs[i:i+1]
                output = self.net(single_input)
                
                # L2 norm of output
                output_norm = output.pow(2).sum()
                output_norm.backward()
                
                # Accumulate absolute gradients
                for name, param in self.net.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        new_omega[name] += param.grad.abs().detach()
                
                samples_processed += 1
            
            if samples_processed >= num_samples:
                break
        
        print(f"[MAS] ✓ Processed {samples_processed} samples")
        
        # Normalize by number of samples
        if samples_processed > 0:
            for name in new_omega:
                new_omega[name] /= samples_processed
        
        # CRITICAL FIX: Normalize omega to sum=1.0 (not sum=num_params!)
        current_sum = sum(o.sum().item() for o in new_omega.values())
        
        if current_sum > 0:
            # Scale so total Omega = 1.0
            scale_factor = 1.0 / current_sum
            for name in new_omega:
                new_omega[name] *= scale_factor
            
            print(f"[MAS] Normalized Omega (sum: {current_sum:.2f} → 1.0)")
        
        # Accumulate omega across tasks
        if not self.omega:
            self.omega = new_omega
            print(f"[MAS] ✓ Initialized Omega (Task 1)")
        else:
            alpha = 1.0 / (self.task_count + 1)
            print(f"[MAS] Accumulating Omega (alpha={alpha:.4f})...")
            for name in self.omega:
                if name in new_omega:
                    self.omega[name] = (1 - alpha) * self.omega[name] + alpha * new_omega[name]
            print(f"[MAS] ✓ Accumulated Omega (Task {self.task_count + 1})")
        
        # Store optimal parameters
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
        """Fallback to uniform omega"""
        print("[MAS] Using uniform Omega as fallback...")
        
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        base_omega = 1.0 / total_params  # Normalize to sum=1.0
        
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
        """Print statistics"""
        print("\n" + "="*70)
        print("[MAS] Omega Statistics:")
        print("="*70)
        
        total_omega = sum(o.sum().item() for o in self.omega.values())
        total_elements = sum(o.numel() for o in self.omega.values())
        avg_omega = total_omega / total_elements if total_elements > 0 else 0
        
        print(f"[MAS]   Total Omega sum: {total_omega:.6f}")
        print(f"[MAS]   Average Omega: {avg_omega:.10f}")
        print(f"[MAS]   Protected parameters: {total_elements:,}")
        print(f"[MAS]   Tasks accumulated: {self.task_count}")
        
        # Per-layer stats
        layer_importance = []
        for name, omega in self.omega.items():
            layer_sum = omega.sum().item()
            layer_avg = omega.mean().item()
            layer_max = omega.max().item()
            layer_importance.append((name, layer_sum, layer_avg, layer_max))
        
        layer_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n[MAS]   Top Important Layers:")
        for i, (name, total, avg, max_val) in enumerate(layer_importance[:5]):
            print(f"[MAS]     {i+1}. {name}")
            print(f"[MAS]        Sum: {total:.6f}, Avg: {avg:.10f}, Max: {max_val:.6f}")
        
        print("="*70 + "\n")
    
    def mas_penalty(self):
        """
        Compute MAS penalty.
        With normalized Omega (sum=1.0), lambda can be higher (10-100)
        """
        if not self.omega or not self.old_params:
            return torch.tensor(0.0).to(self.device)
        
        penalty = torch.tensor(0.0).to(self.device)
        
        for name, param in self.net.named_parameters():
            if name in self.omega and name in self.old_params:
                diff = (param - self.old_params[name]).pow(2)
                penalty += (self.omega[name] * diff).sum()
        
        # NO division by task_count
        # Safety clip (higher max because Omega is normalized)
        penalty = torch.clamp(penalty, max=1000.0)
        
        return penalty
    
    def end_task(self, dataset):
        """Called at end of each task"""
        self.compute_omega(dataset, num_samples=200)