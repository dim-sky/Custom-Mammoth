"""
Frozen Backbone + MLP Classifier + LwF
"""

from models.frozen_backbone_linear_lwf_pretrained import FrozenBackboneLinearLwF


class FrozenBackboneMLPLwF(FrozenBackboneLinearLwF):
    """Frozen backbone with MLP classifier and LwF"""
    NAME = 'frozen_backbone_mlp_lwf_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    @staticmethod
    def get_parser(parser):
        parser = FrozenBackboneLinearLwF.get_parser(parser)
        parser.add_argument('--mlp_hidden_dim', type=int, default=256)
        parser.add_argument('--mlp_dropout', type=float, default=0.5)
        return parser