"""
Implementations of BYOL in PyTorch

Bootstrap Your Own Latent A New Approach to Self-Supervised Learning:
    https://arxiv.org/pdf/2006.07733.pdf
Acknowledgments:
    https://github.com/Lightning-AI/lightning-bolts
    https://github.com/deepmind/deepmind-research/tree/master/byol
"""
from copy import deepcopy
from typing import Any, Union

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch import Tensor, nn


class BYOL(nn.Module):
    def __init__(self, encoder, learning_rate=0.2, weight_decay=1.5e-6, moving_average_decay = 0.99,
                 encoder_out_dim=2048, projector_hidden_dim=4096, projector_out_dim=256):
        """
        Args:
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            encoder_out_dim: output dimension of base_encoder
            projector_hidden_size: hidden layer size of projector MLP
            projector_out_dim: output size of projector MLP
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.online_network = SiameseArm(encoder, encoder_out_dim, projector_hidden_dim, projector_out_dim)
        self.target_network = deepcopy(self.online_network)
        self.predictor = MLP(projector_out_dim, projector_hidden_dim, projector_out_dim)
        self.beta = moving_average_decay
        self.optimizer = self.configure_optimizers()

    def forward(self, img1, img2):
        """Returns the encoded representation of a view.
        Args:
            img1 (Tensor): 
            img1 (Tensor): 
        """
        # Calculate similarity loss in each direction
        loss_12 = self.calculate_loss(img1, img2)
        loss_21 = self.calculate_loss(img2, img1)

        # Calculate total loss
        total_loss = loss_12 + loss_21

        # Log losses
        log = ({"loss_12": loss_12, "loss_21": loss_21})
        
        self.update_moving_average()
        # return 
        return total_loss, log

    def calculate_loss(self, v_online, v_target):
        """Calculates similarity loss between the online network prediction of target network projection.
        Args:
            v_online (Tensor): Online network view
            v_target (Tensor): Target network view
        """
        _, z1 = self.online_network(v_online)
        h1 = self.predictor(z1)
        with torch.no_grad():
            _, z2 = self.target_network(v_target)
        loss = -2 * F.cosine_similarity(h1, z2).mean()
        return loss    

    @torch.no_grad()
    def update_moving_average(self):
        """Update target network parameters."""
        for online_p, target_p in zip(self.online_network.parameters(), self.target_network.parameters()):
            target_p.data = self.beta  * target_p.data + (1.0 - self.beta) * online_p.data


    def configure_optimizers(self):
        optimizer = Adam(list(self.online_network.parameters()) + list(self.predictor.parameters()),
                         lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


class MLP(nn.Module):
    """MLP architecture used as projectors in online and target networks and predictors in the online network.
    Args:
        input_dim (int, optional): Input dimension. Defaults to 2048.
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 4096.
        output_dim (int, optional): Output dimension. Defaults to 256.
    Note:
        Default values for input, hidden, and output dimensions are based on values used in BYOL.
    """

    def __init__(self, input_dim=2048, hidden_dim=4096, output_dim=256):

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, x):
        return self.model(x)


class SiameseArm(nn.Module):
    """SiameseArm consolidates the encoder and projector networks of BYOL's symmetric architecture into a single
    class.
    Args:
        encoder (Union[str, nn.Module], optional): Online and target network encoder architecture.
            Defaults to "resnet50".
        encoder_out_dim (int, optional): Output dimension of encoder. Defaults to 2048.
        projector_hidden_dim (int, optional): Online and target network projector network hidden dimension.
            Defaults to 4096.
        projector_out_dim (int, optional): Online and target network projector network output dimension.
            Defaults to 256.
    """

    def __init__(self, encoder, encoder_out_dim=2048, projector_hidden_dim=4096, projector_out_dim=256):
        super().__init__()
        self.encoder = encoder
        self.projector = MLP(encoder_out_dim, projector_hidden_dim, projector_out_dim)

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        return y, z