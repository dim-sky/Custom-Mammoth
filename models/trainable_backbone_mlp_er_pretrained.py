"""
Trainable Backbone + MLP Classifier + Experience Replay
"""

import torch
from models.trainable_backbone_linear_er_pretrained import TrainableBackboneLinearER


class TrainableBackboneMLPER(TrainableBackboneLinearER):
    """Trainable backbone with MLP classifier and Experience Replay"""
    NAME = 'trainable_backbone_mlp_er_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        print(f"\n{'='*70}")
        print(f"[TrainableER-MLP] Initializing with ER")
        print(f"[TrainableER-MLP] Hidden dim: {getattr(args, 'mlp_hidden_dim', 256)}")
        print(f"[TrainableER-MLP] Dropout: {getattr(args, 'mlp_dropout', 0.5)}")
        print(f"{'='*70}\n")
        
        super().__init__(backbone, loss, args, transform, dataset)
    
    @staticmethod
    def get_parser(parser):
        parser = TrainableBackboneLinearER.get_parser(parser)
        parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                          help='MLP hidden dimension (default: 256)')
        parser.add_argument('--mlp_dropout', type=float, default=0.5,
                          help='MLP dropout rate (default: 0.5)')
        return parser