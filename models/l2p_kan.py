"""
L2P with KAN Classifier
"""

import torch
import torch.nn as nn
from models.l2p import L2P
from utils.args import ArgumentParser


class L2PKAN(L2P):
    """L2P with KAN classifier head"""
    NAME = 'l2p_kan'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser = L2P.get_parser(parser)
        parser.add_argument('--kan_hidden_dim', type=int, default=64)
        parser.add_argument('--kan_num_grids', type=int, default=4)
        parser.add_argument('--kan_grid_min', type=float, default=-2.0)
        parser.add_argument('--kan_grid_max', type=float, default=2.0)
        return parser
    
    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset)
        self._replace_head_with_kan()
    
    def _replace_head_with_kan(self):
        print("\n" + "="*70)
        print("[L2P-KAN] Replacing head with KAN classifier...")
        print("="*70)
        
        try:
            from fastkan import FastKAN
            print("[L2P-KAN] ✓ FastKAN library found")
        except ImportError:
            print("[L2P-KAN] ❌ ERROR: FastKAN not installed!")
            print("[L2P-KAN] Run: pip install fastkan")
            raise
        
        head_replaced = False
        
        if hasattr(self.net, 'model') and hasattr(self.net.model, 'head'):
            old_head = self.net.model.head
            in_features = self._get_head_input_dim(old_head)
            
            hidden_dim = self.args.kan_hidden_dim
            num_grids = self.args.kan_num_grids
            grid_min = self.args.kan_grid_min
            grid_max = self.args.kan_grid_max
            out_features = self.num_classes
            
            layer_dims = [in_features, hidden_dim, out_features]
            
            new_head = FastKAN(
                layers_hidden=layer_dims,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=True,
                spline_weight_init_scale=0.1
            )
            
            self.net.model.head = new_head.to(self.device)
            
            kan_params = sum(p.numel() for p in new_head.parameters())
            print(f"[L2P-KAN] ✓ Replaced: {in_features} → {hidden_dim} → {out_features}")
            print(f"[L2P-KAN] ✓ KAN parameters: {kan_params:,}")
            print(f"[L2P-KAN] ✓ Grids: {num_grids}, Range: [{grid_min}, {grid_max}]")
            head_replaced = True
        
        elif hasattr(self.net, 'head'):
            old_head = self.net.head
            in_features = self._get_head_input_dim(old_head)
            
            hidden_dim = self.args.kan_hidden_dim
            num_grids = self.args.kan_num_grids
            grid_min = self.args.kan_grid_min
            grid_max = self.args.kan_grid_max
            out_features = self.num_classes
            
            layer_dims = [in_features, hidden_dim, out_features]
            
            new_head = FastKAN(
                layers_hidden=layer_dims,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=True,
                spline_weight_init_scale=0.1
            )
            
            self.net.head = new_head.to(self.device)
            
            kan_params = sum(p.numel() for p in new_head.parameters())
            print(f"[L2P-KAN] ✓ Replaced: {in_features} → {hidden_dim} → {out_features}")
            print(f"[L2P-KAN] ✓ KAN parameters: {kan_params:,}")
            head_replaced = True
        
        if not head_replaced:
            print("[L2P-KAN] ⚠️ WARNING: Could not find head!")
        
        print("="*70 + "\n")
    
    def _get_head_input_dim(self, head):
        if hasattr(head, 'in_features'):
            return head.in_features
        if isinstance(head, nn.Sequential) and hasattr(head[0], 'in_features'):
            return head[0].in_features
        if hasattr(head, 'weight'):
            return head.weight.shape[1]
        return 768