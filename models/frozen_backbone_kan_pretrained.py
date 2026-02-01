"""
Frozen Backbone με ImageNet Pre-trained Weights + KAN Classifier

KAN (Kolmogorov-Arnold Networks):
- Αντί για σταθερά weights (Linear/MLP), χρησιμοποιεί learnable basis functions
- Adaptive: Μαθαίνει τις καλύτερες transformations για τα features
- Πιο expressive από Linear, πιο interpretable από MLP
"""

import torch
import torch.nn as nn
from argparse import ArgumentParser

from models.frozen_backbone_pretrained import FrozenBackbonePretrained
from models import register_model


@register_model('frozen_backbone_kan_pretrained')
class FrozenBackboneKANPretrained(FrozenBackbonePretrained):
    """
    Frozen Backbone με ImageNet pre-trained weights + KAN classifier.
    
    Inheritance:
    - FrozenBackbonePretrained: Φορτώνει ImageNet weights + freezes backbone
    - Αυτό το class: Αντικαθιστά τον Linear classifier με KAN
    """
    NAME = 'frozen_backbone_kan_pretrained'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform, dataset):
        """
        Initialization:
        1. Καλεί parent (frozen backbone + ImageNet weights)
        2. Αντικαθιστά Linear classifier με KAN
        3. Unfreeze KAN parameters
        """
        
        # ========== ΒΗΜΑ 1: Καλούμε Parent Init ==========
        # Αυτό κάνει:
        # - Δημιουργεί το network (ResNet18)
        # - Παγώνει το backbone
        # - Φορτώνει ImageNet weights
        # - Δημιουργεί Linear classifier (που θα αντικαταστήσουμε)
        super().__init__(backbone, loss, args, transform, dataset)
        
        # ========== ΒΗΜΑ 2: Αντικατάσταση με KAN ==========
        print("[FrozenBackboneKAN] Replacing Linear classifier with KAN...")
        self._replace_with_kan_classifier()
        
        print(f"[FrozenBackboneKAN] ✓ KAN classifier installed!")
        
        # ========== ΒΗΜΑ 3: Debug Info ==========
        self._print_trainable_params()
    
    def _replace_with_kan_classifier(self):
        """
        Αντικαθιστά τον Linear classifier με KAN.
        
        Steps:
        1. Import FastKAN library
        2. Βρίσκει το feature dimension (512 για ResNet18)
        3. Διαβάζει KAN hyperparameters από args
        4. Δημιουργεί KAN classifier
        5. Unfreeze KAN parameters
        """
        
        # ========== IMPORT FASTKAN ==========
        try:
            # Προσπαθούμε να κάνουμε import το FastKAN
            from fastkan import FastKAN
            print("[FrozenBackboneKAN] ✓ FastKAN library found")
            
        except ImportError:
            # Αν δεν υπάρχει, δίνουμε error με οδηγίες
            print("[ERROR] FastKAN not installed!")
            print("[ERROR] Install with one of:")
            print("[ERROR]   pip install fastkan")
            print("[ERROR]   pip install git+https://github.com/ZiyaoLi/fast-kan.git")
            raise ImportError("FastKAN is required for KAN classifier")
        
        # ========== ΒΡΙΣΚΟΥΜΕ FEATURE DIMENSION ==========
        # Το feature dimension είναι το output του frozen backbone
        # Για ResNet18: 512
        # Για ResNet50: 2048
        
        if hasattr(self.net, 'num_features'):
            # Μερικά networks έχουν το attribute num_features
            feat_dim = self.net.num_features
            
        elif hasattr(self.net.classifier, 'in_features'):
            # Αλλιώς, παίρνουμε από τον υπάρχοντα classifier
            feat_dim = self.net.classifier.in_features
            
        else:
            # Fallback: 512 για ResNet18
            feat_dim = 512
            print(f"[WARNING] Could not detect feature dim, using default: {feat_dim}")
        
        # ========== ΔΙΑΒΑΖΟΥΜΕ KAN HYPERPARAMETERS ==========
        # Αυτά έρχονται από command line args (--kan_hidden_dim, etc.)
        # Αν δεν δοθούν, χρησιμοποιούμε defaults
        
        # Hidden dimension: Μέγεθος κρυφού layer
        # Default: 64 (μικρότερο από MLP για ταχύτητα)
        kan_hidden_dim = getattr(self.args, 'kan_hidden_dim', 64)
        
        # Grid size: Πόσα control points για τα splines
        # Default: 5 (περισσότερα = πιο expressive, αλλά πιο αργό)
        kan_grid_size = getattr(self.args, 'kan_grid_size', 5)
        
        # Spline order: Polynomial degree των splines
        # Default: 3 (cubic splines - smooth & expressive)
        kan_spline_order = getattr(self.args, 'kan_spline_order', 3)
        
        # ========== PRINT ARCHITECTURE INFO ==========
        print(f"[FrozenBackboneKAN] KAN architecture:")
        print(f"  Input dimension:  {feat_dim}")       # From frozen backbone
        print(f"  Hidden dimension: {kan_hidden_dim}")  # User-specified
        print(f"  Grid size:        {kan_grid_size}")   # For splines
        print(f"  Spline order:     {kan_spline_order}") # Polynomial degree
        print(f"  Output dimension: {self.num_classes}") # Number of classes (10)
        
        # ========== ΔΗΜΙΟΥΡΓΙΑ KAN CLASSIFIER ==========
        # Architecture: Input → Hidden → Output
        # Όπως MLP, αλλά με learnable basis functions αντί για ReLU
        
        # Δημιουργούμε λίστα με dimensions για κάθε layer
        # [512, 64, 10] σημαίνει:
        # - Layer 1: 512 → 64 (με KAN basis functions)
        # - Layer 2: 64 → 10 (με KAN basis functions)
        layer_dims = [feat_dim, kan_hidden_dim, self.num_classes]
        
        # Δημιουργούμε το KAN network
        self.net.classifier = FastKAN(
            layers_hidden=layer_dims,      # [512, 64, 10]
            grid_size=kan_grid_size,       # 5 control points
            spline_order=kan_spline_order, # Cubic splines (degree 3)
        )
        
        # ========== UNFREEZE KAN PARAMETERS ==========
        # Το backbone είναι frozen, αλλά ο KAN classifier πρέπει να είναι trainable
        for param in self.net.classifier.parameters():
            param.requires_grad = True  # Κάνε trainable
    
    def _print_trainable_params(self):
        """
        Debug function: Εκτυπώνει πληροφορίες για trainable parameters.
        
        Βοηθάει να επιβεβαιώσουμε ότι:
        - Backbone είναι frozen (requires_grad=False)
        - Classifier είναι trainable (requires_grad=True)
        """
        
        # ========== ΜΕΤΡΑΜΕ PARAMETERS ==========
        # Χωρίζουμε σε trainable και frozen
        
        trainable = 0  # Counter για trainable params
        total = 0      # Counter για total params
        
        # Loop όλα τα parameters του network
        for param in self.net.parameters():
            param_count = param.numel()  # Αριθμός στοιχείων (elements)
            total += param_count
            
            if param.requires_grad:
                # Αυτό το parameter είναι trainable
                trainable += param_count
        
        # Frozen parameters = Total - Trainable
        frozen = total - trainable
        
        # ========== ΕΚΤΥΠΩΣΗ STATISTICS ==========
        percentage = 100.0 * trainable / total if total > 0 else 0
        
        print(f"[FrozenBackboneKAN] Parameter Statistics:")
        print(f"  Trainable params: {trainable:,} ({percentage:.2f}%)")
        print(f"  Frozen params:    {frozen:,} ({100-percentage:.2f}%)")
        print(f"  Total params:     {total:,}")
        
        # ========== ΕΠΙΒΕΒΑΙΩΣΗ ==========
        # Θέλουμε ~99% frozen (backbone), ~1% trainable (KAN)
        if percentage < 5.0:
            print(f"  ✓ Backbone properly frozen (only {percentage:.2f}% trainable)")
        else:
            print(f"  ⚠️ WARNING: Too many trainable params ({percentage:.2f}%)")
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Προσθέτει KAN-specific command line arguments.
        
        Αυτά επιτρέπουν στον χρήστη να customize το KAN architecture:
        - --kan_hidden_dim: Μέγεθος hidden layer
        - --kan_grid_size: Πόσα control points για splines
        - --kan_spline_order: Polynomial degree
        """
        
        # Hidden dimension (default: 64)
        # Μικρότερο από MLP (256) για ταχύτητα
        parser.add_argument(
            '--kan_hidden_dim', 
            type=int, 
            default=64,
            help='Hidden dimension for KAN classifier (default: 64)'
        )
        
        # Grid size (default: 5)
        # Περισσότερα grid points = πιο expressive, αλλά πιο αργό
        parser.add_argument(
            '--kan_grid_size', 
            type=int, 
            default=5,
            help='Grid size for KAN splines (default: 5)'
        )
        
        # Spline order (default: 3 = cubic)
        # 1 = linear, 2 = quadratic, 3 = cubic (recommended)
        parser.add_argument(
            '--kan_spline_order', 
            type=int, 
            default=3,
            help='Spline order for KAN (default: 3, cubic)'
        )
        
        return parser


# ============================================================
# TI EINAI TO KAN? (Conceptual Explanation)
# ============================================================
"""
Linear Classifier:
    y = W · x + b
    - Σταθερά weights W
    - Linear transformation

MLP Classifier:
    h = ReLU(W₁ · x + b₁)
    y = W₂ · h + b₂
    - Σταθερό activation (ReLU)
    - Non-linear αλλά fixed function

KAN Classifier:
    y = Σᵢ φᵢ(x)
    όπου φᵢ(x) = learnable basis function
    
    - Κάθε φᵢ είναι spline function με learnable coefficients
    - Adaptive: Μαθαίνει τις best transformations για τα data
    - Πιο flexible από Linear/MLP
    
Γιατί KAN για Continual Learning?
1. Smooth basis functions → λιγότερο forgetting
2. Local adaptivity → μπορεί να adapt σε νέα tasks χωρίς να καταστρέψει τα παλιά
3. Expressive → καλύτερη accuracy
"""