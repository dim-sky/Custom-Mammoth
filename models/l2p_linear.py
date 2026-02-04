"""
L2P with Linear Classifier
Replaces L2P's default head with simple Linear classifier
"""

import torch
import torch.nn as nn
from models.l2p import L2P
from utils.args import ArgumentParser


class L2PLinear(L2P):
    """L2P with Linear classifier head"""
    NAME = 'l2p_linear'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    def __init__(self, backbone, loss, args, transform, dataset=None):
        # Let L2P create everything
        super().__init__(backbone, loss, args, transform, dataset)
        
        # Replace head after initialization
        self._replace_head_with_linear()
    
    def _replace_head_with_linear(self):
        """
        Replace L2P head with Linear classifier.
        Searches for head in model structure and replaces it.
        """
        print("\n" + "="*70)
        print("[L2P-Linear] Attempting to replace head with Linear classifier...")
        print("="*70)
        
        # Search for head in different possible locations
        head_replaced = False
        
        # Try: self.net.model.head
        if hasattr(self.net, 'model') and hasattr(self.net.model, 'head'):
            old_head = self.net.model.head
            print(f"[L2P-Linear] Found head at self.net.model.head")
            print(f"[L2P-Linear] Old head type: {type(old_head)}")
            
            # Determine input dimension
            in_features = self._get_head_input_dim(old_head)
            out_features = self.num_classes
            
            # Create and replace
            new_head = nn.Linear(in_features, out_features)
            self.net.model.head = new_head.to(self.device)
            
            print(f"[L2P-Linear] ✓ Replaced head: {in_features} → {out_features}")
            head_replaced = True
        
        # Try: self.net.head
        elif hasattr(self.net, 'head'):
            old_head = self.net.head
            print(f"[L2P-Linear] Found head at self.net.head")
            print(f"[L2P-Linear] Old head type: {type(old_head)}")
            
            in_features = self._get_head_input_dim(old_head)
            out_features = self.num_classes
            
            new_head = nn.Linear(in_features, out_features)
            self.net.head = new_head.to(self.device)
            
            print(f"[L2P-Linear] ✓ Replaced head: {in_features} → {out_features}")
            head_replaced = True
        
        if not head_replaced:
            print("[L2P-Linear] ⚠️ WARNING: Could not find head to replace!")
            print("[L2P-Linear] Printing model structure for debugging:")
            print(self.net)
        
        print("="*70 + "\n")
    
    def _get_head_input_dim(self, head):
        """Extract input dimension from existing head"""
        # Try Linear layer
        if hasattr(head, 'in_features'):
            return head.in_features
        
        # Try Sequential (get first layer)
        if isinstance(head, nn.Sequential):
            first_layer = head[0]
            if hasattr(first_layer, 'in_features'):
                return first_layer.in_features
        
        # Try weight shape
        if hasattr(head, 'weight'):
            return head.weight.shape[1]
        
        # Default for ViT-B/16
        print("[L2P-Linear] Using default dimension: 768")
        return 768