"""
Trainable Backbone + KAC Classifier + LwF (Learning without Forgetting)
"""

import torch
from argparse import Namespace

from models.trainable_backbone_linear_lwf_pretrained import TrainableBackboneLinearLwF
from utils.args import ArgumentParser


class TrainableBackboneKACLwF(TrainableBackboneLinearLwF):
    """Trainable backbone with KAC classifier and LwF"""
    NAME = 'trainable_backbone_kac_lwf_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone: torch.nn.Module, loss: torch.nn.Module,
                 args: Namespace, transform: torch.nn.Module, dataset):
        
        print(f"\n{'='*70}")
        print(f"[Trainable-LwF-KAC] Initializing")
        print(f"[Trainable-LwF-KAC] Num grids: {getattr(args, 'kac_num_grids', 16)}")
        print(f"{'='*70}\n")
        
        super().__init__(backbone, loss, args, transform, dataset)
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser = TrainableBackboneLinearLwF.get_parser(parser)
        # KAC args inherited from baseline
        return parser
    