"""
Frozen Backbone με ImageNet Pre-trained Weights + FastKAN Classifier
WORKING VERSION - Correct FastKAN API
"""

import torch
import torch.nn as nn
from argparse import ArgumentParser

from models.frozen_backbone_pretrained import FrozenBackbonePretrained
from models import register_model


@register_model('frozen_backbone_kan_pretrained')
class FrozenBackboneKANPretrained(FrozenBackbonePretrained):
    """Frozen Backbone + FastKAN Classifier"""
    NAME = 'frozen_backbone_kan_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        # Parent init (frozen backbone + ImageNet weights)
        super().__init__(backbone, loss, args, transform, dataset)
        
        # Replace Linear with FastKAN
        print("[FrozenBackboneKAN] Replacing Linear classifier with FastKAN...")
        self._replace_with_kan_classifier()
        print(f"[FrozenBackboneKAN] ✓ FastKAN classifier installed!")
        
        self._print_trainable_params()
    
    def _replace_with_kan_classifier(self):
        """Replace Linear classifier with FastKAN"""
        
        # Import FastKAN
        try:
            from fastkan import FastKAN
            print("[FrozenBackboneKAN] ✓ FastKAN library found")
        except ImportError:
            print("[ERROR] FastKAN not installed!")
            print("[ERROR] Run: pip install fastkan")
            raise
        
        # Get feature dimension
        if hasattr(self.net, 'num_features'):
            feat_dim = self.net.num_features
        elif hasattr(self.net.classifier, 'in_features'):
            feat_dim = self.net.classifier.in_features
        else:
            feat_dim = 512
        
        # Get KAN hyperparameters
        kan_hidden_dim = getattr(self.args, 'kan_hidden_dim', 64)
        num_grids = getattr(self.args, 'kan_num_grids', 8)  # Correct parameter name!
        grid_min = getattr(self.args, 'kan_grid_min', -2.0)
        grid_max = getattr(self.args, 'kan_grid_max', 2.0)
        
        print(f"[FrozenBackboneKAN] KAN architecture:")
        print(f"  Input dimension:  {feat_dim}")
        print(f"  Hidden dimension: {kan_hidden_dim}")
        print(f"  Output dimension: {self.num_classes}")
        print(f"  Num grids:        {num_grids}")
        print(f"  Grid range:       [{grid_min}, {grid_max}]")
        
        # Create FastKAN with correct API
        layer_dims = [feat_dim, kan_hidden_dim, self.num_classes]
        
        self.net.classifier = FastKAN(
            layers_hidden=layer_dims,  # [512, 64, 10]
            grid_min=grid_min,          # -2.0
            grid_max=grid_max,          # 2.0
            num_grids=num_grids,        # 8 (not grid_size!)
            use_base_update=True,       # Default
            spline_weight_init_scale=0.1  # Default
        )
        
        # Unfreeze KAN parameters
        for param in self.net.classifier.parameters():
            param.requires_grad = True
        
        print(f"[FrozenBackboneKAN] ✓ FastKAN created successfully")
    
    def _print_trainable_params(self):
        """Print parameter statistics"""
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.net.parameters())
        percentage = 100.0 * trainable / total
        
        print(f"[FrozenBackboneKAN] Parameter Statistics:")
        print(f"  Trainable: {trainable:,} ({percentage:.2f}%)")
        print(f"  Total:     {total:,}")
        
        if percentage < 5.0:
            print(f"  ✓ Backbone properly frozen")
        else:
            print(f"  ⚠️ Too many trainable params")
    
    @staticmethod
    def get_parser(parser):
        """Add FastKAN-specific arguments"""
        parser.add_argument('--kan_hidden_dim', type=int, default=64,
                          help='Hidden dimension for KAN (default: 64)')
        parser.add_argument('--kan_num_grids', type=int, default=8,
                          help='Number of grid points for splines (default: 8)')
        parser.add_argument('--kan_grid_min', type=float, default=-2.0,
                          help='Min value for grid range (default: -2.0)')
        parser.add_argument('--kan_grid_max', type=float, default=2.0,
                          help='Max value for grid range (default: 2.0)')
        return parser