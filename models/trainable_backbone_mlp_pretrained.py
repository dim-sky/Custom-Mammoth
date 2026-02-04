"""
MLP Classifier with TRAINABLE Backbone
Uses differential learning rates for backbone vs classifier
"""

import torch
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace


class TrainableBackboneMLPPretrained(ContinualModel):
    """MLP classifier with trainable pretrained backbone"""
    NAME = 'trainable_backbone_mlp_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone: torch.nn.Module, loss: torch.nn.Module,
                 args: Namespace, transform: torch.nn.Module, dataset: ContinualDataset):
        
        super().__init__(backbone, loss, args, transform, dataset)
        
        self._set_all_trainable()
        self._verify_trainable()
        self._print_trainable_summary()
        self._setup_differential_lr()
    
    def _set_all_trainable(self):
        """Ensure all parameters are trainable"""
        print("\n" + "="*70)
        print("[TRAINABLE] Setting all parameters to trainable...")
        print("="*70)
        
        trainable_count = 0
        
        for name, param in self.net.named_parameters():
            param.requires_grad = True
            trainable_count += param.numel()
        
        print(f"[TRAINABLE] ✓ All parameters trainable: {trainable_count:,}")
        print("="*70 + "\n")
    
    def _verify_trainable(self):
        """Verify all parameters are trainable"""
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        
        print("\n" + "="*70)
        print("[VERIFY] Checking trainable status...")
        print("="*70)
        print(f"[VERIFY] Total parameters: {total:,}")
        print(f"[VERIFY] Trainable parameters: {trainable:,}")
        print(f"[VERIFY] Percentage trainable: {100 * trainable / total:.2f}%")
        
        if trainable == total:
            print(f"[VERIFY] ✓ SUCCESS: All parameters are trainable")
        else:
            print(f"[VERIFY] ⚠️ WARNING: {total - trainable:,} parameters are frozen!")
        print("="*70 + "\n")
    
    def _print_trainable_summary(self):
        """Print parameter groups"""
        print("\n" + "="*70)
        print("[SUMMARY] Parameter Groups:")
        print("="*70)
        
        backbone_params = 0
        classifier_params = 0
        
        for name, param in self.net.named_parameters():
            if 'classifier' in name:
                classifier_params += param.numel()
            else:
                backbone_params += param.numel()
        
        print(f"[SUMMARY]   Backbone parameters: {backbone_params:,}")
        print(f"[SUMMARY]   Classifier parameters: {classifier_params:,}")
        print(f"[SUMMARY]   Total: {backbone_params + classifier_params:,}")
        print("="*70 + "\n")
    
    def _setup_differential_lr(self):
        """Setup optimizer with differential learning rates"""
        backbone_lr = getattr(self.args, 'backbone_lr', self.args.lr * 0.1)
        classifier_lr = self.args.lr
        
        print("\n" + "="*70)
        print("[OPTIMIZER] Setting up differential learning rates...")
        print("="*70)
        print(f"[OPTIMIZER]   Backbone LR: {backbone_lr:.6f}")
        print(f"[OPTIMIZER]   Classifier LR: {classifier_lr:.6f}")
        print(f"[OPTIMIZER]   Ratio: {classifier_lr/backbone_lr:.1f}x")
        print("="*70 + "\n")
        
        backbone_params = []
        classifier_params = []
        
        for name, param in self.net.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': classifier_params, 'lr': classifier_lr}
        ]
        
        if self.args.optimizer == 'sgd':
            self.opt = torch.optim.SGD(
                param_groups,
                momentum=self.args.optim_mom,
                weight_decay=self.args.optim_wd,
                nesterov=self.args.optim_nesterov
            )
        elif self.args.optimizer == 'adam':
            self.opt = torch.optim.Adam(
                param_groups,
                weight_decay=self.args.optim_wd
            )
        elif self.args.optimizer == 'adamw':
            self.opt = torch.optim.AdamW(
                param_groups,
                weight_decay=self.args.optim_wd
            )
        
        print(f"[OPTIMIZER] ✓ Optimizer created with {len(param_groups)} parameter groups")
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """Standard training step"""
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()
        return loss.item()
    
    @staticmethod
    def get_parser(parser):
        parser.add_argument('--backbone_lr', type=float, default=None,
                          help='Learning rate for backbone (default: 0.1 * lr)')
        parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                          help='Hidden dimension for MLP classifier (default: 256)')
        parser.add_argument('--mlp_dropout', type=float, default=0.5,
                          help='Dropout rate for MLP classifier (default: 0.5)')
        return parser