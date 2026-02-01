"""
Frozen Backbone με ImageNet Pre-trained Weights + MLP Classifier + Experience Replay
Συνδυάζει: Frozen features + MLP capacity + Memory replay
"""
import torch
import torch.nn as nn
from argparse import ArgumentParser

from models.frozen_backbone_er_pretrained import FrozenBackboneERPretrained
from models import register_model


@register_model('frozen_backbone_mlp_er_pretrained')
class FrozenBackboneMLPERPretrained(FrozenBackboneERPretrained):
    """
    Frozen Backbone + MLP Classifier + Experience Replay.
    Combines frozen pretrained features with MLP classifier and memory replay.
    """
    NAME = 'frozen_backbone_mlp_er_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        # Call parent (sets up frozen backbone + ImageNet weights + ER buffer)
        super().__init__(backbone, loss, args, transform, dataset)
        
        # Replace Linear classifier with MLP
        self._replace_with_mlp_classifier()
        
        print(f"[FrozenMLPER] MLP Classifier + ER ready!")
    
    def _replace_with_mlp_classifier(self):
        """Αντικαθιστά τον Linear classifier με MLP."""
        
        # Get feature dimension
        if hasattr(self.net, 'num_features'):
            feat_dim = self.net.num_features
        elif hasattr(self.net.classifier, 'in_features'):
            feat_dim = self.net.classifier.in_features
        else:
            feat_dim = 512  # default
        
        # Get MLP hyperparameters
        hidden_dim = getattr(self.args, 'mlp_hidden_dim', 256)
        dropout = getattr(self.args, 'mlp_dropout', 0.5)
        
        print(f"[FrozenMLPER] Replacing Linear with MLP...")
        print(f"[FrozenMLPER]   Input: {feat_dim}")
        print(f"[FrozenMLPER]   Hidden: {hidden_dim}")
        print(f"[FrozenMLPER]   Dropout: {dropout}")
        print(f"[FrozenMLPER]   Output: {self.num_classes}")
        
        # Create MLP classifier
        self.net.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes)
        )
        
        # Unfreeze MLP parameters
        for param in self.net.classifier.parameters():
            param.requires_grad = True
        
        # Print trainable params
        trainable = sum(p.numel() for p in self.net.classifier.parameters())
        print(f"[FrozenMLPER] Trainable MLP params: {trainable:,}")
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """Add MLP-specific arguments."""
        # Get ER arguments from parent
        parser = FrozenBackboneERPretrained.get_parser(parser)
        
        # Add MLP arguments
        parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                          help='Hidden dimension for MLP classifier (default: 256)')
        parser.add_argument('--mlp_dropout', type=float, default=0.5,
                          help='Dropout rate for MLP classifier (default: 0.5)')
        return parser