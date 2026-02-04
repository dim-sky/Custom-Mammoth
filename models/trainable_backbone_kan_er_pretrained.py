"""
Trainable Backbone + KAN Classifier + Experience Replay
"""

import torch
from models.trainable_backbone_linear_er_pretrained import TrainableBackboneLinearER


class TrainableBackboneKANER(TrainableBackboneLinearER):
    """Trainable backbone with KAN classifier and Experience Replay"""
    NAME = 'trainable_backbone_kan_er_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        print(f"\n{'='*70}")
        print(f"[TrainableER-KAN] Initializing with ER")
        print(f"[TrainableER-KAN] Hidden dim: {getattr(args, 'kan_hidden_dim', 64)}")
        print(f"[TrainableER-KAN] Num grids: {getattr(args, 'kan_num_grids', 8)}")
        print(f"{'='*70}\n")
        
        super().__init__(backbone, loss, args, transform, dataset)
    
    @staticmethod
    def get_parser(parser):
        parser = TrainableBackboneLinearER.get_parser(parser)
        parser.add_argument('--kan_hidden_dim', type=int, default=64,
                          help='KAN hidden dimension (default: 64)')
        parser.add_argument('--kan_num_grids', type=int, default=8,
                          help='Number of grid points (default: 8)')
        parser.add_argument('--kan_grid_min', type=float, default=-2.0,
                          help='Grid minimum (default: -2.0)')
        parser.add_argument('--kan_grid_max', type=float, default=2.0,
                          help='Grid maximum (default: 2.0)')
        return parser