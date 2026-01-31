"""
Frozen Backbone με CIFAR Pre-trained Weights
Χρησιμοποιεί την pytorch-cifar-models library
"""
from argparse import ArgumentParser
from models.frozen_backbone import FrozenBackbone
from models import register_model


@register_model('frozen_backbone_cifar_pretrained')
class FrozenBackboneCIFARPretrained(FrozenBackbone):
    NAME = 'frozen_backbone_cifar_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        print("\n" + "="*70)
        print("[FrozenBackboneCIFARPretrained] Using CIFAR PRE-TRAINED weights")
        print("="*70 + "\n")
        
        super().__init__(backbone, loss, args, transform, dataset)
        
        print("[FrozenBackboneCIFARPretrained] Loading CIFAR weights...")
        self._load_cifar_weights()
        print("[FrozenBackboneCIFARPretrained] ✓ Weights loaded!\n")
    
    def _load_cifar_weights(self):
        try:
            import pytorch_cifar_models as cifar_models
            
            backbone_name = getattr(self.args, 'backbone', 'resnet18')
            
            if 'resnet20' in backbone_name.lower():
                pretrained_model = cifar_models.cifar10_resnet20(pretrained=True)
            elif 'resnet32' in backbone_name.lower():
                pretrained_model = cifar_models.cifar10_resnet32(pretrained=True)
            elif 'resnet56' in backbone_name.lower():
                pretrained_model = cifar_models.cifar10_resnet56(pretrained=True)
            else:
                print(f"[WARNING] No CIFAR weights for {backbone_name}")
                return
            
            # Copy weights
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.net.state_dict()
            
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                              if k in model_dict and 'fc' not in k and 'classifier' not in k}
            
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict, strict=False)
            
            print(f"[FrozenBackboneCIFARPretrained] ✓ Copied {len(pretrained_dict)} layers")
            
        except ImportError:
            print("[ERROR] pytorch-cifar-models not installed!")
            print("[ERROR] Run: pip install pytorch-cifar-models")
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        return parser