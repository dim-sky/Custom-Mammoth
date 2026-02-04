"""
L2P with MLP Classifier
"""

import torch
import torch.nn as nn
from models.l2p import L2P
from utils.args import ArgumentParser


class L2PMLP(L2P):
    """L2P with MLP classifier head"""
    NAME = 'l2p_mlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser = L2P.get_parser(parser)
        parser.add_argument('--mlp_hidden_dim', type=int, default=256)
        parser.add_argument('--mlp_dropout', type=float, default=0.5)
        return parser
    
    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset)
        self._replace_head_with_mlp()
    
    def _replace_head_with_mlp(self):
        print("\n" + "="*70)
        print("[L2P-MLP] Replacing head with MLP classifier...")
        print("="*70)
        
        head_replaced = False
        
        if hasattr(self.net, 'model') and hasattr(self.net.model, 'head'):
            old_head = self.net.model.head
            in_features = self._get_head_input_dim(old_head)
            
            hidden_dim = self.args.mlp_hidden_dim
            dropout = self.args.mlp_dropout
            out_features = self.num_classes
            
            new_head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_features)
            )
            
            self.net.model.head = new_head.to(self.device)
            
            mlp_params = sum(p.numel() for p in new_head.parameters())
            print(f"[L2P-MLP] ✓ Replaced: {in_features} → {hidden_dim} → {out_features}")
            print(f"[L2P-MLP] ✓ MLP parameters: {mlp_params:,}")
            head_replaced = True
        
        elif hasattr(self.net, 'head'):
            old_head = self.net.head
            in_features = self._get_head_input_dim(old_head)
            
            hidden_dim = self.args.mlp_hidden_dim
            dropout = self.args.mlp_dropout
            out_features = self.num_classes
            
            new_head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_features)
            )
            
            self.net.head = new_head.to(self.device)
            
            mlp_params = sum(p.numel() for p in new_head.parameters())
            print(f"[L2P-MLP] ✓ Replaced: {in_features} → {hidden_dim} → {out_features}")
            print(f"[L2P-MLP] ✓ MLP parameters: {mlp_params:,}")
            head_replaced = True
        
        if not head_replaced:
            print("[L2P-MLP] ⚠️ WARNING: Could not find head!")
        
        print("="*70 + "\n")
    
    def _get_head_input_dim(self, head):
        if hasattr(head, 'in_features'):
            return head.in_features
        if isinstance(head, nn.Sequential) and hasattr(head[0], 'in_features'):
            return head[0].in_features
        if hasattr(head, 'weight'):
            return head.weight.shape[1]
        return 768