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
        # Αυτό θα κάνει freeze το backbone και setup το classifier
        super().__init__(backbone, loss, args, transform, dataset)
        
        # ΜΕΤΑ το setup, φορτώνουμε pre-trained weights
        print("[FrozenBackbonePretrained] Loading pre-trained weights...")
        self._load_pretrained_weights()
        print("[FrozenBackbonePretrained] ✓ Pre-trained weights loaded!\n")
    
    def _load_pretrained_weights(self):
        """
        Φορτώνει ImageNet pre-trained weights από torchvision.
        """
        backbone_name = getattr(self.args, 'backbone', 'resnet18')
        
        print(f"[FrozenBackbonePretrained] Backbone type: {backbone_name}")
        
        # ==================== RESNET FAMILY ====================
        if 'resnet18' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading ResNet18 (ImageNet)...")
            pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
            
        elif 'resnet34' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading ResNet34 (ImageNet)...")
            pretrained_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
            
        elif 'resnet50' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading ResNet50 (ImageNet)...")
            pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        # ==================== ΝΕΑ: ResNet101, ResNet152 ====================
        elif 'resnet101' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading ResNet101 (ImageNet)...")
            pretrained_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        elif 'resnet152' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading ResNet152 (ImageNet)...")
            pretrained_model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        # ==================== ΝΕΑ: EFFICIENTNET FAMILY ====================
        elif 'efficientnet_b0' in backbone_name.lower() or backbone_name == 'efficientnet':
            print(f"[FrozenBackbonePretrained] Downloading EfficientNet-B0 (ImageNet)...")
            pretrained_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        elif 'efficientnet_b1' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading EfficientNet-B1 (ImageNet)...")
            pretrained_model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        elif 'efficientnet_b2' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading EfficientNet-B2 (ImageNet)...")
            pretrained_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        # ==================== ΝΕΑ: MOBILENET FAMILY ====================
        elif 'mobilenet_v2' in backbone_name.lower() or 'mobilenetv2' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading MobileNetV2 (ImageNet)...")
            pretrained_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        elif 'mobilenet_v3_small' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading MobileNetV3-Small (ImageNet)...")
            pretrained_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        elif 'mobilenet_v3_large' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading MobileNetV3-Large (ImageNet)...")
            pretrained_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        # ==================== ΝΕΑ: VGG FAMILY ====================
        elif 'vgg16' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading VGG16-BN (ImageNet)...")
            pretrained_model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        elif 'vgg19' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading VGG19-BN (ImageNet)...")
            pretrained_model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        # ==================== ΝΕΑ: DENSENET FAMILY ====================
        elif 'densenet121' in backbone_name.lower():
            print(f"[FrozenBackbonePretrained] Downloading DenseNet121 (ImageNet)...")
            pretrained_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self._copy_weights(pretrained_model)
        
        # ==================== FALLBACK ====================
        elif 'resnet32' in backbone_name.lower():
            print(f"[WARNING] ResNet32 is CIFAR-specific - no ImageNet weights available")
            print(f"[WARNING] Keeping random initialization")
        
        else:
            print(f"[WARNING] No pre-trained weights available for {backbone_name}")
            print(f"[WARNING] Keeping random initialization")
    
    def _copy_weights(self, pretrained_model):
        """
        Αντιγράφει weights από το torchvision pre-trained model.
        Αγνοεί το final classifier layer (διαφορετικό output size).
        """
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.net.state_dict()
        
        # Κρατάμε μόνο τα layers που υπάρχουν και στα δύο models
        # Εξαιρούμε το fc/classifier layer
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'fc' not in k and 'classifier' not in k}
        
        # Ενημέρωση
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict, strict=False)
        
        print(f"[FrozenBackbonePretrained] ✓ Copied {len(pretrained_dict)} layers from ImageNet model")
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """No additional arguments needed"""
        return parser