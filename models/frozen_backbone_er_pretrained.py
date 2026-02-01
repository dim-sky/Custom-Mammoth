"""
Frozen Backbone με ImageNet Pre-trained Weights + Experience Replay
Συνδυάζει frozen features με replay strategy για καλύτερη stability
"""
import torch
from argparse import ArgumentParser

from models.frozen_backbone_pretrained import FrozenBackbonePretrained
from models import register_model
from utils.buffer import Buffer


@register_model('frozen_backbone_er_pretrained')
class FrozenBackboneERPretrained(FrozenBackbonePretrained):
    """
    Frozen Backbone + Experience Replay.
    Frozen features για stability + Memory replay για plasticity.
    """
    NAME = 'frozen_backbone_er_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        # Καλεί parent (frozen backbone με ImageNet weights)
        super().__init__(backbone, loss, args, transform, dataset)
        
        # Προσθήκη replay buffer
        self.buffer = Buffer(self.args.buffer_size, self.device)
        
        print(f"[FrozenBackboneER] Experience Replay enabled")
        print(f"[FrozenBackboneER] Buffer size: {self.args.buffer_size}")
    
    def observe(self, inputs, labels, not_aug_inputs):
        """
        Training step με experience replay.
        """
        self.opt.zero_grad()
        
        # ========== Current Task Data ==========
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        
        # ========== Replay από Buffer ==========
        if not self.buffer.is_empty():
            # Get replay batch
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            
            # Forward pass on replay data
            buf_outputs = self.net(buf_inputs)
            loss += self.loss(buf_outputs, buf_labels)
        
        # ========== Backward Pass ==========
        # Μόνο ο classifier ενημερώνεται (backbone frozen)
        loss.backward()
        self.opt.step()
        
        # ========== Update Buffer ==========
        # Προσθήκη νέων examples στο buffer
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)
        
        return loss.item()
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """Προσθέτει ER-specific arguments."""
        parser.add_argument('--buffer_size', type=int, default=500,
                          help='Size of the replay buffer (default: 500)')
        parser.add_argument('--minibatch_size', type=int, default=32,
                          help='Minibatch size for replay (default: 32)')
        return parser