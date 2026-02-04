"""
Frozen Backbone + KAC Classifier + LwF
"""

from models.frozen_backbone_linear_lwf_pretrained import FrozenBackboneLinearLwF


class FrozenBackboneKACLwF(FrozenBackboneLinearLwF):
    """Frozen backbone with KAC classifier and LwF"""
    NAME = 'frozen_backbone_kac_lwf_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    @staticmethod
    def get_parser(parser):
        parser = FrozenBackboneLinearLwF.get_parser(parser)
        # KAC args inherited from baseline
        return parser