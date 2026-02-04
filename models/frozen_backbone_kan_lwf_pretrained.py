"""
Frozen Backbone + KAN Classifier + LwF
"""

from models.frozen_backbone_linear_lwf_pretrained import FrozenBackboneLinearLwF


class FrozenBackboneKANLwF(FrozenBackboneLinearLwF):
    """Frozen backbone with KAN classifier and LwF"""
    NAME = 'frozen_backbone_kan_lwf_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    @staticmethod
    def get_parser(parser):
        parser = FrozenBackboneLinearLwF.get_parser(parser)
        parser.add_argument('--kan_hidden_dim', type=int, default=64)
        parser.add_argument('--kan_num_grids', type=int, default=8)
        parser.add_argument('--kan_grid_min', type=float, default=-2.0)
        parser.add_argument('--kan_grid_max', type=float, default=2.0)
        return parser