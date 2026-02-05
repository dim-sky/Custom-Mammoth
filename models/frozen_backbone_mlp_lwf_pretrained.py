"""
Frozen Backbone + MLP Classifier + LwF
Inherits from MLP baseline and adds LwF functionality
"""

import torch
import torch.nn.functional as F
from copy import deepcopy
from models.frozen_backbone_mlp_pretrained import FrozenBackboneMLPPretrained


class FrozenBackboneMLPLwF(FrozenBackboneMLPPretrained):
    """Frozen backbone with MLP classifier and LwF"""
    NAME = 'frozen_backbone_mlp_lwf_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        # Call MLP parent to get MLP classifier
        super().__init__(backbone, loss, args, transform, dataset)
        
        # Add LwF-specific attributes
        self.old_net = None
        self.eye = torch.eye(self.num_classes).to(self.device)
        
        print(f"\n{'='*70}")
        print(f"[LwF-MLP] Learning without Forgetting enabled")
        print(f"[LwF-MLP] Classifier type: {type(self.net.classifier)}")
        print(f"{'='*70}\n")
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """Training with LwF distillation"""
        if self.current_task > 0 and self.old_net is not None:
            with torch.no_grad():
                old_logits = torch.sigmoid(self.old_net(inputs))
        else:
            old_logits = None
        
        self.opt.zero_grad()
        loss = self.get_loss(inputs, labels, self.current_task, old_logits)
        loss.backward()
        self.opt.step()
        
        return loss.item()
    
    def get_loss(self, inputs, labels, task_idx, old_logits):
        """Compute LwF loss"""
        classes_per_task = self.num_classes // self.n_tasks
        pc = task_idx * classes_per_task
        ac = (task_idx + 1) * classes_per_task
        
        outputs = self.net(inputs)[:, :ac]
        
        if task_idx == 0:
            targets = self.eye[labels][:, :ac]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
        else:
            targets = self.eye[labels][:, pc:ac]
            comb_targets = torch.cat((old_logits[:, :pc], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
        
        return loss
    
    def end_task(self, dataset):
        """Save model copy"""
        self.old_net = deepcopy(self.net)
        self.old_net.eval()
        for param in self.old_net.parameters():
            param.requires_grad = False
        self.net.train()