"""
Frozen Backbone με ImageNet Pre-trained Weights + FastKAN + Experience Replay

Combines:
- Frozen pretrained backbone (stability)
- FastKAN classifier (adaptive B-splines)
- Experience Replay (memory buffer)
"""

import torch
from argparse import ArgumentParser

from models.frozen_backbone_kan_pretrained import FrozenBackboneKANPretrained
from models import register_model
from utils.buffer import Buffer


@register_model('frozen_backbone_kan_er_pretrained')
class FrozenBackboneKANERPretrained(FrozenBackboneKANPretrained):
    """
    Frozen Backbone + FastKAN + Experience Replay.
    
    Inherits:
        - FrozenBackboneKANPretrained: Frozen backbone + KAN classifier
        - Adds: Experience Replay buffer
    """
    NAME = 'frozen_backbone_kan_er_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        # Initialize parent (frozen backbone + KAN classifier)
        super().__init__(backbone, loss, args, transform, dataset)
        
        # Add replay buffer
        self.buffer = Buffer(self.args.buffer_size, self.device)
        
        print(f"[FrozenKANER] Experience Replay enabled")
        print(f"[FrozenKANER] Buffer size: {self.args.buffer_size}")
    
    def observe(self, inputs, labels, not_aug_inputs):
        """
        Training step with experience replay.
        
        Args:
            inputs: Current batch images
            labels: Current batch labels
            not_aug_inputs: Non-augmented images (for buffer storage)
        
        Returns:
            loss: Training loss value
        """
        self.opt.zero_grad()
        
        # ========== Forward pass on current data ==========
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        
        # ========== Experience Replay ==========
        if not self.buffer.is_empty():
            # Sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            
            # Forward pass on replayed data
            buf_outputs = self.net(buf_inputs)
            loss += self.loss(buf_outputs, buf_labels)
        
        # ========== Backward pass ==========
        # Only KAN classifier parameters are updated (backbone frozen)
        loss.backward()
        self.opt.step()
        
        # ========== Update buffer ==========
        # Add new examples to buffer (reservoir sampling)
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)
        
        return loss.item()
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """Add ER-specific arguments"""
        # Get KAN arguments from parent
        parser = FrozenBackboneKANPretrained.get_parser(parser)
        
        # Add ER arguments
        parser.add_argument(
            '--buffer_size', 
            type=int, 
            default=500,
            help='Size of the replay buffer (default: 500)'
        )
        parser.add_argument(
            '--minibatch_size', 
            type=int, 
            default=32,
            help='Minibatch size for replay (default: 32)'
        )
        
        return parser