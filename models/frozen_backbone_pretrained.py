"""
Frozen Backbone με ImageNet Pre-trained Weights
Επεκτείνει το FrozenBackbone για να φορτώνει pre-trained weights από torchvision
"""
import torch
import torch.nn as nn
from torchvision import models
from argparse import ArgumentParser

from models.frozen_backbone import FrozenBackbone
from models import register_model


@register_model('frozen_backbone_pretrained')
class FrozenBackbonePretrained(FrozenBackbone):
    """
    Frozen Backbone με ImageNet pre-trained weights.
    Κληρονομεί από FrozenBackbone και προσθέτει pre-training loading.
    """
    NAME = 'frozen_backbone_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        print("\n" + "="*70)
        print("[FrozenBackbonePretrained] Using ImageNet PRE-TRAINED weights")
        print("="*70 + "\n")
        
        # Καλεί τον parent __init__ (frozen_backbone)
        super().__init__(backbone, loss, args, transform, dataset)
        
        # Φορτώνουμε pre-trained weights
        print("[FrozenBackbonePretrained] Loading pre-trained weights...")
        self._load_pretrained_weights()
        print("[FrozenBackbonePretrained] ✓ Pre-trained weights loaded!\n")
    
    def _load_pretrained_weights(self):
        """Φορτώνει ImageNet pre-trained weights από torchvision."""
        backbone_name = getattr(self.args, 'backbone', 'resnet18')
        print(f"[FrozenBackbonePretrained] Backbone type: {backbone_name}")
        
        # ResNet family
        if 'resnet18' in backbone_name.lower():
            pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
            
        elif 'resnet34' in backbone_name.lower():
            pretrained_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
            
        elif 'resnet50' in backbone_name.lower():
            pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        elif 'resnet32' in backbone_name.lower():
            print(f"[WARNING] ResNet32 is CIFAR-specific - no ImageNet weights available")
            print(f"[WARNING] Keeping random initialization")
        
        else:
            print(f"[WARNING] No pre-trained weights for {backbone_name}")
            print(f"[WARNING] Keeping random initialization")
    
    def _copy_weights(self, pretrained_model):
        """Αντιγράφει weights από το torchvision pre-trained model."""
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.net.state_dict()
        
        # Κρατάμε μόνο layers που υπάρχουν και στα δύο (εξαιρούμε classifier)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'fc' not in k and 'classifier' not in k}
        
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict, strict=False)
        
        print(f"[FrozenBackbonePretrained] ✓ Copied {len(pretrained_dict)} layers")
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        return parser
