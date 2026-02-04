"""
Trainable Backbone + KAC Classifier + Experience Replay
"""

import torch
from models.trainable_backbone_linear_er_pretrained import TrainableBackboneLinearER


class TrainableBackboneKACER(TrainableBackboneLinearER):
    """Trainable backbone with KAC classifier and Experience Replay"""
    NAME = 'trainable_backbone_kac_er_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        print(f"\n{'='*70}")
        print(f"[TrainableER-KAC] Initializing with ER")
        print(f"[TrainableER-KAC] Num grids: {getattr(args, 'kac_num_grids', 16)}")
        print(f"{'='*70}\n")
        
        super().__init__(backbone, loss, args, transform, dataset)
    
    @staticmethod
    def get_parser(parser):
        parser = TrainableBackboneLinearER.get_parser(parser)
        # KAC-specific arguments inherited from baseline KAC model
        return parser