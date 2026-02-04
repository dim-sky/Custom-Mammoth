"""
Sequential TinyImageNet resized to 224x224.

This dataset variant is intended for Vision Transformer (ViT)-based models
(e.g. L2P), which require 224x224 input resolution.

It should NOT be used with ResNet-style backbones that expect 32x32 inputs.
"""

import torchvision.transforms as transforms

from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.utils import set_default_from_args


class SequentialTinyImagenet224(SequentialTinyImagenet):
    """
    Sequential TinyImageNet dataset resized to 224x224.

    This dataset is compatible with ViT-based continual learning methods.
    """

    NAME = 'seq-tinyimg-224'

    # Mean and std from TinyImageNet
    MEAN = [0.4807, 0.4485, 0.3980]
    STD = [0.2541, 0.2456, 0.2604]

    # Training transform: resize + data augmentation
    TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Test transform: deterministic resize
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        # Default number of epochs for TinyImageNet
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    @set_default_from_args('backbone')
    def get_backbone():
        # Default backbone (ignored by L2P, which uses ViT internally)
        return 'vit'
