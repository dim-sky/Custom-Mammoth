"""
Frozen Backbone με ImageNet Pre-trained Weights + MLP Classifier
"""
import torch
import torch.nn as nn
from argparse import ArgumentParser

from models.frozen_backbone_pretrained import FrozenBackbonePretrained
from models import register_model


@register_model('frozen_backbone_mlp_pretrained')
class FrozenBackboneMLPPretrained(FrozenBackbonePretrained):
    """
    Frozen Backbone με ImageNet pre-trained weights + MLP classifier.
    """
    NAME = 'frozen_backbone_mlp_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        # Καλεί τον parent __init__ (φορτώνει ImageNet weights, freezes backbone)
        super().__init__(backbone, loss, args, transform, dataset)
        
        # OVERRIDE: Αντικαθιστούμε τον Linear classifier με MLP
        self._replace_with_mlp_classifier()
        
        print(f"[FrozenBackboneMLPPretrained] MLP classifier installed!")
        self._print_trainable_params()
    
    def _replace_with_mlp_classifier(self):
        """Αντικαθιστά τον linear classifier με MLP."""
        
        # Βρες feature dimension
        if hasattr(self.net, 'num_features'):
            feat_dim = self.net.num_features
        elif hasattr(self.net, 'classifier') and hasattr(self.net.classifier, 'in_features'):
            feat_dim = self.net.classifier.in_features
        else:
            feat_dim = 512  # default
        
        # Διάβασε hyperparameters από args (ή defaults)
        hidden_dim = getattr(self.args, 'mlp_hidden_dim', 256)
        dropout = getattr(self.args, 'mlp_dropout', 0.5)
        
        print(f"[FrozenBackboneMLPPretrained] MLP architecture:")
        print(f"  Input: {feat_dim}")
        print(f"  Hidden: {hidden_dim}")
        print(f"  Dropout: {dropout}")
        print(f"  Output: {self.num_classes}")
        
        # Δημιούργησε MLP classifier
        self.net.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes)
        )
        
        # Unfreeze MLP parameters
        for param in self.net.classifier.parameters():
            param.requires_grad = True
    
    def _print_trainable_params(self):
        """Debug info για trainable parameters."""
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.net.parameters())
        percentage = 100.0 * trainable / total
        print(f"[FrozenBackboneMLPPretrained] Trainable: {trainable:,} / Total: {total:,} ({percentage:.2f}%)")
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """Προσθέτει MLP-specific arguments."""
        parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                          help='Hidden dimension for MLP classifier (default: 256)')
        parser.add_argument('--mlp_dropout', type=float, default=0.5,
                          help='Dropout rate for MLP classifier (default: 0.5)')
        return parser