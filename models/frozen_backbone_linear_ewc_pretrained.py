"""
Frozen Backbone + Linear Classifier + EWC
"""

import torch
from argparse import ArgumentParser

from models.frozen_backbone_pretrained import FrozenBackbonePretrained
from models.utils.ewc_mixin import EWCMixin
from models import register_model


@register_model('frozen_backbone_linear_ewc_pretrained')
class FrozenBackboneLinearEWCPretrained(EWCMixin, FrozenBackbonePretrained):
    """
    Frozen Backbone + Linear Classifier + EWC regularization.
    """
    NAME = 'frozen_backbone_linear_ewc_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        # Initialize EWC mixin
        EWCMixin.__init__(self)
        
        # Initialize parent (frozen backbone + Linear classifier)
        FrozenBackbonePretrained.__init__(self, backbone, loss, args, transform, dataset)
        
        # Set EWC lambda from args
        self.ewc_lambda = getattr(self.args, 'ewc_lambda', 1000.0)
        
        print(f"[LinearEWC] EWC regularization enabled")
        print(f"[LinearEWC] Lambda: {self.ewc_lambda}")
    
    def observe(self, inputs, labels, not_aug_inputs):
        """Training step με EWC penalty"""
        self.opt.zero_grad()
        
        # Forward pass
        outputs = self.net(inputs)
        
        # Standard loss
        ce_loss = self.loss(outputs, labels)
        
        # EWC penalty
        ewc_loss = self.ewc_penalty()
        
        # Total loss
        total_loss = ce_loss + self.ewc_lambda * ewc_loss
        
        # Backward
        total_loss.backward()
        self.opt.step()
        
        return total_loss.item()
    
    def end_task(self, dataset):
        """Called after each task"""
        # Compute Fisher for current task
        EWCMixin.end_task(self, dataset)
    
    @staticmethod
    def get_parser(parser):
        parser.add_argument('--ewc_lambda', type=float, default=1000.0,
                          help='EWC regularization strength (default: 1000)')
        return parser
    










   