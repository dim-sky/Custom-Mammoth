"""
Frozen Backbone + KAN Classifier + EWC
"""

from argparse import ArgumentParser

from models.frozen_backbone_kan_pretrained import FrozenBackboneKANPretrained
from models.utils.ewc_mixin import EWCMixin
from models import register_model


@register_model('frozen_backbone_kan_ewc_pretrained')
class FrozenBackboneKANEWCPretrained(EWCMixin, FrozenBackboneKANPretrained):
    """Frozen Backbone + FastKAN + EWC"""
    NAME = 'frozen_backbone_kan_ewc_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        EWCMixin.__init__(self)
        FrozenBackboneKANPretrained.__init__(self, backbone, loss, args, transform, dataset)
        
        self.ewc_lambda = getattr(self.args, 'ewc_lambda', 1000.0)
        print(f"[KANEWC] EWC lambda: {self.ewc_lambda}")
    
    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        
        outputs = self.net(inputs)
        ce_loss = self.loss(outputs, labels)
        ewc_loss = self.ewc_penalty()
        total_loss = ce_loss + self.ewc_lambda * ewc_loss
        
        total_loss.backward()
        self.opt.step()
        
        return total_loss.item()
    
    def end_task(self, dataset):
        EWCMixin.end_task(self, dataset)
    
    @staticmethod
    def get_parser(parser):
        parser = FrozenBackboneKANPretrained.get_parser(parser)
        parser.add_argument('--ewc_lambda', type=float, default=1000.0)
        return parser