"""
Linear Classifier with MAS - Frozen Backbone
"""

import torch
from models.utils.continual_model import ContinualModel
from models.utils.mas_mixin import MASMixin
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace


class LinearMAS(ContinualModel, MASMixin):
    """Linear classifier with MAS regularization"""
    NAME = 'frozen_backbone_linear_mas_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone: torch.nn.Module, loss: torch.nn.Module,
                 args: Namespace, transform: torch.nn.Module, dataset: ContinualDataset):
        
        super().__init__(backbone, loss, args, transform, dataset)
        self.mas_lambda = args.mas_lambda if hasattr(args, 'mas_lambda') else 1.0
        
        print(f"\n[LinearMAS] MAS regularization enabled")
        print(f"[LinearMAS] Lambda: {self.mas_lambda}")
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """Training step with MAS penalty"""
        self.opt.zero_grad()
        
        outputs = self.net(inputs)
        ce_loss = self.loss(outputs, labels)
        mas_loss = self.mas_penalty()
        total_loss = ce_loss + self.mas_lambda * mas_loss
        
        total_loss.backward()
        self.opt.step()
        
        return total_loss.item()
    
    def end_task(self, dataset):
        """Called at the end of each task"""
        MASMixin.end_task(self, dataset)
    
    @staticmethod
    def get_parser(parser):
        parser.add_argument('--mas_lambda', type=float, default=1.0,
                          help='MAS regularization strength (default: 1.0)')
        return parser