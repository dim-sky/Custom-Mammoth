"""
L2P with KAC Classifier
"""

import torch
import torch.nn as nn
from models.l2p import L2P
from utils.args import ArgumentParser


class L2PKAC(L2P):
    """L2P with KAC classifier head"""
    NAME = 'l2p_kac'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser = L2P.get_parser(parser)
        parser.add_argument('--kac_num_grids', type=int, default=4)
        parser.add_argument('--kac_grid_min', type=float, default=-2.0)
        parser.add_argument('--kac_grid_max', type=float, default=2.0)
        return parser
    
    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset)
        self._replace_head_with_kac()
    
    def _replace_head_with_kac(self):
        print("\n" + "="*70)
        print("[L2P-KAC] Replacing head with KAC classifier...")
        print("="*70)
        
        # For now, use Linear as placeholder
        # Replace with actual KAC when you have the implementation
        print("[L2P-KAC] ⚠️ Using Linear as placeholder for KAC")
        print("[L2P-KAC] TODO: Replace with actual KAC implementation")
        
        head_replaced = False
        
        if hasattr(self.net, 'model') and hasattr(self.net.model, 'head'):
            old_head = self.net.model.head
            in_features = self._get_head_input_dim(old_head)
            out_features = self.num_classes
            
            # TODO: Replace with KAC
            new_head = nn.Linear(in_features, out_features)
            
            self.net.model.head = new_head.to(self.device)
            
            print(f"[L2P-KAC] ✓ Replaced: {in_features} → {out_features}")
            head_replaced = True
        
        elif hasattr(self.net, 'head'):
            old_head = self.net.head
            in_features = self._get_head_input_dim(old_head)
            out_features = self.num_classes
            
            new_head = nn.Linear(in_features, out_features)
            
            self.net.head = new_head.to(self.device)
            
            print(f"[L2P-KAC] ✓ Replaced: {in_features} → {out_features}")
            head_replaced = True
        
        if not head_replaced:
            print("[L2P-KAC] ⚠️ WARNING: Could not find head!")
        
        print("="*70 + "\n")
    
    def _get_head_input_dim(self, head):
        if hasattr(head, 'in_features'):
            return head.in_features
        if isinstance(head, nn.Sequential) and hasattr(head[0], 'in_features'):
            return head[0].in_features
        if hasattr(head, 'weight'):
            return head.weight.shape[1]
        return 768
    
