import torch
import torch.nn
from backbone  import get_backbone ## Φέρνει το pretrained backbone (πχ ResNet18)
from models.utils.continual_model import ContinualModel # Βασική κλάση από όπου κληρονομούν όλα τα cl models
from argparse import ArgumentParser # για comandline ορίσματα
from utils.args import add_management_args, add_experiment_args
from models import register_model



@register_model('frozen_backbone')  # <-- Αυτό κάνει τη μαγεία!
class FrozenBackbone(ContinualModel):

    # Από την κλάσση ContinualModel κληρονωμούμε (optimizer, loss, function etc.)

    NAME: str = 'frozen_backbone'
    COMPATIBILITY = ['class-il', 'task-il']


    def __init__(self, backbone, loss, args, transform, dataset):
        super(FrozenBackbone, self).__init__(backbone, loss, args, transform, dataset)
        # backbone: Το neural network architecture (πχ ResNet18)  | self.net
        # loss: Η loss function (πχ crossentropy loss)
        # args: Τα command-line arguments ([πχ lr,epochs κλπ)
 

        # Παίρνω όλα τα weights/biases του network και τα "παγώνω"
        for param in self.net.parameters():
            param.requires_grad = False


        # Βρίσκω το feature dimension του backbone
        # Το feature dimension είναι η διάσταση του feature vector που παράγει το backbone μετά το convolutional μέρος και πριν τον τελικό classifier.
        # Διαφορετικά backbones αποθηκεύουν το feature dimension διαφορετικά
        # --> κάποια έχουν num_features attribute (MobileNet)
        # --> κάποια έχουν classifier.in_features (π.χ. ResNet)
        # --> Αν δεν βρψ τίποτα θεωρώ το 512
        if hasattr(self.net, 'num_features'):
            feat_dim = self.net.num_features
        elif hasattr(self.net, 'classifier'):
            feat_dim = self.net.classifier.in_features
        else:
            feat_dim = 512  # default για ResNet18


        # Αντικαθιστώ το παλιό classifier με καινούριο linear layer
        self.net.classifier = torch.nn.Linear(feat_dim, self.num_classes)

        for param in self.net.classifier.parameters():
            param.requires_grad = True

        # Debug info
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.net.parameters())
        print(f"[FrozenBackbone] Trainable: {trainable:,} / Total: {total:,}")
            

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        
        loss.backward()
        self.opt.step()
        
        return loss.item() 
        
        
    @staticmethod
    def get_parser():
        parser = ArgumentParser(description='Frozen Backbone with Linear Head')
        add_management_args(parser)
        add_experiment_args(parser)
        return parser


