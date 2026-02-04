# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega,
# Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sequential TinyImageNet resized to 224x224 (ViT-friendly).

Why this file exists:
- The original `SequentialTinyImagenet` builds its own test transform (ToTensor + Normalize)
  and therefore keeps images at their native resolution (64x64).
- ViT/L2P expects 224x224 inputs.
- So we create a new dataset variant that forces BOTH train and test to use 224x224 transforms
  by overriding `get_data_loaders()`.

Use with:
  --dataset seq-tinyimg-224
"""

from typing import Tuple

import torch
import torchvision.transforms as transforms

from datasets.seq_tinyimagenet import SequentialTinyImagenet, TinyImagenet, MyTinyImagenet
from datasets.utils.continual_dataset import store_masked_loaders
from datasets.utils import set_default_from_args
from utils.conf import base_path


class SequentialTinyImagenet224(SequentialTinyImagenet):
    """
    Sequential TinyImageNet dataset resized to 224x224.
    Suitable for ViT-based continual learning methods (e.g. L2P).
    """

    NAME = 'seq-tinyimg-224'

    # Keep TinyImageNet statistics (can also use ImageNet stats; these are fine for consistency)
    MEAN = [0.4807, 0.4485, 0.3980]
    STD = [0.2541, 0.2456, 0.2604]

    # Train transform: augmentation + resize/crop to 224
    TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Test transform: deterministic resize/crop to 224
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        IMPORTANT:
        Override parent implementation so that the test loader also uses self.TEST_TRANSFORM
        (otherwise it stays 64x64 and ViT crashes).
        """
        train_dataset = MyTinyImagenet(
            base_path() + 'TINYIMG',
            train=True,
            download=True,
            transform=self.TRANSFORM
        )

        test_dataset = TinyImagenet(
            base_path() + 'TINYIMG',
            train=False,
            download=True,
            transform=self.TEST_TRANSFORM
        )

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    @set_default_from_args("backbone")
    def get_backbone():
        # L2P ignores this (it constructs its own ViT backbone),
        # but keep 'vit' as the default for clarity.
        return "vit"
