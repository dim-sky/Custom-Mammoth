"""
Frozen Backbone με ImageNet Pre-trained Weights + KAC Classifier (RBF-based)

KAC = Kolmogorov-Arnold Classifier
Paper: "KAC: Kolmogorov-Arnold Classifier for Continual Learning" (CVPR 2025 Highlight)
Authors: Yusong Hu, Zichen Liang, et al.

Uses Radial Basis Functions (RBFs) as learnable basis functions.
Official implementation adapted from: https://github.com/Ethanhuhuhu/KAC
"""

import torch
import torch.nn as nn
from argparse import ArgumentParser

from models.frozen_backbone_pretrained import FrozenBackbonePretrained
from models import register_model


# ============================================================
# Official KAC Components (RBF-based)
# ============================================================

class RadialBasisFunction(nn.Module):
    """
    Radial Basis Function layer.
    
    Computes: RBF(x) = exp(-((x - center) / denominator)^2)
    
    Args:
        grid_min: Minimum value for RBF centers
        grid_max: Maximum value for RBF centers
        num_grids: Number of RBF centers
        denominator: Width of RBF kernels (controls smoothness)
    """
    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 16,
        denominator: float = None,
    ):
        super().__init__()
        
        # Create evenly-spaced RBF centers
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        
        # Set RBF width (if not specified, use grid spacing)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, features]
        Returns:
            RBF activations [batch, features, num_grids]
        """
        # Compute distance to each RBF center
        # x[..., None] expands to [batch, features, 1]
        # self.grid is [num_grids]
        # Result: [batch, features, num_grids]
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class KACLayer(nn.Module):
    """
    KAC Layer - Single layer KAC classifier using RBFs.
    
    Architecture:
        Input → LayerNorm → RBF → Linear → (+ Base Linear) → Output
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension (number of classes)
        grid_min: Min value for RBF grid
        grid_max: Max value for RBF grid
        num_grids: Number of RBF centers
        use_base_update: Whether to add residual linear connection
        spline_weight_init_scale: Initialization scale for weights
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 16,
        use_base_update: bool = True,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        
        # Normalize inputs
        self.layernorm = nn.LayerNorm(input_dim)
        
        # RBF basis functions
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        
        # Learnable basis weights (not used in forward, kept for compatibility)
        self.basis_linear = nn.Parameter(torch.zeros([input_dim, num_grids]))
        nn.init.trunc_normal_(self.basis_linear, mean=0, std=spline_weight_init_scale)
        
        # Main transformation: RBF features → output
        self.spline_linear = nn.Linear(input_dim * num_grids, output_dim, bias=False)
        nn.init.zeros_(self.spline_linear.weight)
        
        # Optional residual connection (improves training stability)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features [batch, input_dim]
        Returns:
            Output logits [batch, output_dim]
        """
        # Apply RBF basis functions
        # Shape: [batch, input_dim] → [batch, input_dim, num_grids]
        spline_basis = self.rbf(self.layernorm(x))
        
        # Flatten RBF features
        # Shape: [batch, input_dim, num_grids] → [batch, input_dim * num_grids]
        spline_basis_flat = spline_basis.view(x.shape[0], -1)
        
        # Apply linear transformation
        # Shape: [batch, input_dim * num_grids] → [batch, output_dim]
        output = self.spline_linear(spline_basis_flat)
        
        # Add residual connection if enabled
        if self.use_base_update:
            output = output + self.base_linear(x)
        
        return output


# ============================================================
# Frozen Backbone + KAC Integration
# ============================================================

@register_model('frozen_backbone_kac_pretrained')
class FrozenBackboneKACPretrained(FrozenBackbonePretrained):
    """
    Frozen Backbone with KAC Classifier.
    
    Combines:
    - Frozen ImageNet pre-trained backbone (ResNet18)
    - KAC classifier with RBF basis functions
    
    Reference:
        Hu et al. "KAC: Kolmogorov-Arnold Classifier for Continual Learning"
        CVPR 2025 Highlight
    """
    NAME = 'frozen_backbone_kac_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        # Initialize parent (frozen backbone + ImageNet weights)
        super().__init__(backbone, loss, args, transform, dataset)
        
        # Replace Linear classifier with KAC
        print("[FrozenBackboneKAC] Installing KAC classifier (RBF-based)...")
        self._replace_with_kac_classifier()
        print(f"[FrozenBackboneKAC] ✓ KAC classifier installed!")
        
        # Print parameter statistics
        self._print_trainable_params()
    
    def _replace_with_kac_classifier(self):
        """Replace Linear classifier with official KACLayer"""
        
        # Detect feature dimension from frozen backbone
        if hasattr(self.net, 'num_features'):
            feat_dim = self.net.num_features
        elif hasattr(self.net.classifier, 'in_features'):
            feat_dim = self.net.classifier.in_features
        else:
            feat_dim = 512  # Default for ResNet18
        
        # Get KAC hyperparameters from args
        num_grids = getattr(self.args, 'kac_num_grids', 16)
        grid_min = getattr(self.args, 'kac_grid_min', -2.0)
        grid_max = getattr(self.args, 'kac_grid_max', 2.0)
        
        # use_base_update is ALWAYS True (as per original paper)
        # Residual connection improves training stability and performance
        use_base_update = True
        
        # Print architecture info
        print(f"[FrozenBackboneKAC] Architecture:")
        print(f"  Backbone:       Frozen (ImageNet pre-trained)")
        print(f"  Input dim:      {feat_dim}")
        print(f"  Output dim:     {self.num_classes}")
        print(f"  Basis:          RBF (Radial Basis Functions)")
        print(f"  Num RBFs:       {num_grids}")
        print(f"  RBF range:      [{grid_min}, {grid_max}]")
        print(f"  Base update:    {use_base_update} (always enabled)")
        
        # Create KAC classifier
        self.net.classifier = KACLayer(
            input_dim=feat_dim,
            output_dim=self.num_classes,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=use_base_update,
            spline_weight_init_scale=0.1
        )
        
        # Unfreeze KAC parameters (backbone stays frozen)
        for param in self.net.classifier.parameters():
            param.requires_grad = True
    
    def _print_trainable_params(self):
        """Print trainable vs frozen parameter counts"""
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.net.parameters())
        percentage = 100.0 * trainable / total
        
        print(f"[FrozenBackboneKAC] Parameters:")
        print(f"  Trainable: {trainable:,} ({percentage:.2f}%)")
        print(f"  Frozen:    {total - trainable:,} ({100 - percentage:.2f}%)")
        print(f"  Total:     {total:,}")
        
        if percentage < 5.0:
            print(f"  ✓ Backbone properly frozen")
        else:
            print(f"  ⚠️ Warning: More than 5% trainable")
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """Add KAC-specific command line arguments"""
        
        # Number of RBF centers
        parser.add_argument(
            '--kac_num_grids', 
            type=int, 
            default=16,
            help='Number of RBF centers (default: 16, paper uses 16)'
        )
        
        # RBF grid minimum value
        parser.add_argument(
            '--kac_grid_min', 
            type=float, 
            default=-2.0,
            help='Minimum RBF center value (default: -2.0)'
        )
        
        # RBF grid maximum value
        parser.add_argument(
            '--kac_grid_max', 
            type=float, 
            default=2.0,
            help='Maximum RBF center value (default: 2.0)'
        )
        
        # NOTE: use_base_update is hardcoded to True
        # This follows the original paper and improves training stability
        # No need for command line argument
        
        return parser
    





