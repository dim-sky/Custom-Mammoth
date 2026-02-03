"""
MLP Classifier with MAS - Frozen Backbone (FIXED)
- Replaces the classifier with an actual MLP head
- Freezes ONLY the backbone, keeps the whole head trainable
- Verifies trainable params are within expected range
"""

import torch
import torch.nn as nn
from argparse import Namespace

from models.utils.continual_model import ContinualModel
from models.utils.mas_mixin import MASMixin
from datasets.utils.continual_dataset import ContinualDataset


class MLPMAS(ContinualModel, MASMixin):
    """MLP classifier with MAS regularization and frozen backbone"""
    NAME = 'frozen_backbone_mlp_mas_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self,
                 backbone: torch.nn.Module,
                 loss: torch.nn.Module,
                 args: Namespace,
                 transform: torch.nn.Module,
                 dataset: ContinualDataset):

        # Init base ContinualModel
        super().__init__(backbone, loss, args, transform, dataset)

        # MAS state
        self.omega = {}
        self.old_params = {}
        self.task_count = 0
        self.mas_lambda = float(getattr(args, 'mas_lambda', 1.0))

        # Build/replace classifier with MLP
        self.mlp_hidden_dim = int(getattr(args, 'mlp_hidden_dim', 256))
        self.mlp_dropout = float(getattr(args, 'mlp_dropout', 0.5))
        self._replace_classifier_with_mlp()

        # Freeze backbone (keep head trainable)
        self._freeze_backbone_keep_head_trainable()
        self._verify_trainables()
        self._print_trainable_summary()

        print(f"\n{'='*70}")
        print(f"[MLPMAS] MAS regularization enabled")
        print(f"[MLPMAS] Lambda: {self.mas_lambda}")
        print(f"[MLPMAS] MLP hidden_dim: {self.mlp_hidden_dim}, dropout: {self.mlp_dropout}")
        print(f"{'='*70}\n")

    # --------------------------
    # Model surgery: MLP head
    # --------------------------
    def _infer_feat_dim_and_num_classes(self):
        """
        Tries to infer feature dimension and number of classes from existing classifier.
        Assumes original classifier is Linear(in_features -> num_classes).
        """
        if not hasattr(self.net, 'classifier'):
            raise AttributeError("self.net has no attribute 'classifier'. Cannot infer head dimensions.")

        old_head = self.net.classifier

        # Common case: old classifier is nn.Linear
        if isinstance(old_head, nn.Linear):
            feat_dim = old_head.in_features
            num_classes = old_head.out_features
            return feat_dim, num_classes

        # If it's Sequential or something else, try to find last Linear
        last_linear = None
        for m in old_head.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise RuntimeError("Could not find a Linear layer inside existing classifier to infer dimensions.")

        # feat_dim: best guess from first Linear we find (closest to input)
        first_linear = None
        for m in old_head.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break

        feat_dim = first_linear.in_features if first_linear is not None else last_linear.in_features
        num_classes = last_linear.out_features
        return feat_dim, num_classes

    def _replace_classifier_with_mlp(self):
        feat_dim, num_classes = self._infer_feat_dim_and_num_classes()

        self.net.classifier = nn.Sequential(
            nn.Linear(feat_dim, self.mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.mlp_dropout),
            nn.Linear(self.mlp_hidden_dim, num_classes),
        )

        print("\n" + "="*70)
        print("[HEAD] Replaced classifier with MLP head")
        print("="*70)
        print(f"[HEAD] Feature dim: {feat_dim}")
        print(f"[HEAD] Hidden dim:  {self.mlp_hidden_dim}")
        print(f"[HEAD] Num classes: {num_classes}")
        print("="*70 + "\n")

    # --------------------------
    # Freezing logic
    # --------------------------
    def _freeze_backbone_keep_head_trainable(self):
        """
        Freeze everything except parameters inside self.net.classifier (the head).
        This is robust regardless of layer names (fc1/fc2/head/etc.) as long as they live in .classifier.
        """
        print("\n" + "="*70)
        print("[FREEZE] Freezing backbone (keeping head trainable)...")
        print("="*70)

        # First freeze all
        for p in self.net.parameters():
            p.requires_grad = False

        # Then unfreeze head
        if not hasattr(self.net, 'classifier'):
            raise AttributeError("self.net has no attribute 'classifier'. Cannot unfreeze head.")
        for p in self.net.classifier.parameters():
            p.requires_grad = True

        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        frozen = total - trainable

        print(f"[FREEZE] ✓ Frozen parameters:   {frozen:,}")
        print(f"[FREEZE] ✓ Trainable parameters:{trainable:,}")
        print("="*70 + "\n")

    def _verify_trainables(self):
        """
        Verify that the head is trainable and backbone is frozen.
        We estimate expected trainable params for:
          MLP: (feat_dim*H + H) + (H*C + C)
        where feat_dim inferred from old head, H=mlp_hidden_dim, C=num_classes
        """
        feat_dim, num_classes = self._infer_feat_dim_and_num_classes()
        H = self.mlp_hidden_dim
        C = num_classes

        expected_trainable = (feat_dim * H + H) + (H * C + C)

        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        frozen = total - trainable

        print("\n" + "="*70)
        print("[VERIFY] Checking freeze/trainable status...")
        print("="*70)
        print(f"[VERIFY] Total parameters:     {total:,}")
        print(f"[VERIFY] Trainable parameters: {trainable:,}")
        print(f"[VERIFY] Frozen parameters:    {frozen:,}")
        print(f"[VERIFY] Expected trainable:   ~{expected_trainable:,}")
        print(f"[VERIFY] Percentage frozen:    {100 * frozen / total:.2f}%")

        # Tight-ish bounds to catch "still linear" or "too much unfrozen"
        low = int(expected_trainable * 0.8)
        high = int(expected_trainable * 1.2)

        if trainable < low:
            raise RuntimeError(
                f"[VERIFY] ❌ Too FEW trainable parameters! Expected ~{expected_trainable:,}, got {trainable:,}. "
                f"Likely the MLP head is not actually trainable / not installed."
            )
        if trainable > high:
            raise RuntimeError(
                f"[VERIFY] ❌ Too MANY trainable parameters! Expected ~{expected_trainable:,}, got {trainable:,}. "
                f"Likely some backbone layers are still unfrozen."
            )

        print("[VERIFY] ✓ SUCCESS: Backbone frozen, MLP head trainable within expected range")
        print("="*70 + "\n")

    def _print_trainable_summary(self):
        print("\n" + "="*70)
        print("[SUMMARY] Trainable Layers:")
        print("="*70)
        found = False
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                found = True
                print(f"[SUMMARY]   ✓ {name}: {param.numel():,} params")
        if not found:
            print("[SUMMARY]   ❌ No trainable layers found!")
        print("="*70 + "\n")

    # --------------------------
    # Training step
    # --------------------------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()

        outputs = self.net(inputs)
        ce_loss = self.loss(outputs, labels)

        mas_loss = self.mas_penalty()
        total_loss = ce_loss + self.mas_lambda * mas_loss

        total_loss.backward()
        self.opt.step()

        return total_loss.item()

    def end_task(self, dataset):
        MASMixin.end_task(self, dataset)

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--mas_lambda', type=float, default=1.0,
                            help='MAS regularization strength (try 0.1, 1, 10, 50, 100)')
        parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                            help='Hidden dimension for MLP classifier (default: 256)')
        parser.add_argument('--mlp_dropout', type=float, default=0.5,
                            help='Dropout rate for MLP classifier (default: 0.5)')
        return parser
