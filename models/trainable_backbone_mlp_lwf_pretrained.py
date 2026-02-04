"""
Trainable Backbone + MLP Classifier + LwF (Learning without Forgetting)
"""

import torch
from argparse import Namespace

from models.trainable_backbone_linear_lwf_pretrained import TrainableBackboneLinearLwF
from utils.args import ArgumentParser


class TrainableBackboneMLPLwF(TrainableBackboneLinearLwF):
    """Trainable backbone with MLP classifier and LwF"""
    NAME = 'trainable_backbone_mlp_lwf_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone: torch.nn.Module, loss: torch.nn.Module,
                 args: Namespace, transform: torch.nn.Module, dataset):
        
        print(f"\n{'='*70}")
        print(f"[Trainable-LwF-MLP] Initializing")
        print(f"[Trainable-LwF-MLP] Hidden dim: {getattr(args, 'mlp_hidden_dim', 256)}")
        print(f"[Trainable-LwF-MLP] Dropout: {getattr(args, 'mlp_dropout', 0.5)}")
        print(f"{'='*70}\n")
        
        super().__init__(backbone, loss, args, transform, dataset)
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser = TrainableBackboneLinearLwF.get_parser(parser)
        parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                          help='MLP hidden dimension (default: 256)')
        parser.add_argument('--mlp_dropout', type=float, default=0.5,
                          help='MLP dropout rate (default: 0.5)')
        return parser