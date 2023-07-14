from torch import nn
import torch.nn.functional as F


class SiameseArm(nn.Module):
    def __init__(self, ssl_method,  encoder, encoder_out_dim=2048, projector_hidden_dim=4096, projector_out_dim=256):
        super().__init__()
        self.ssl_method = ssl_method
        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = self.mlp(encoder_out_dim, projector_hidden_dim, projector_out_dim)
        # Predictor
        self.predictor = self.mlp(projector_out_dim, projector_hidden_dim, projector_out_dim)

    
    def mlp(self, in_dim=2048, hidden_dim=4096, out_dim=256):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
    
    def forward(self, x):
        y = self.encoder(x)[0]
        z = self.projector(y)
        h = self.predictor(z)
        if self.ssl_method == 'simclr':
            return z
        elif self.ssl_method == 'byol':
            return y, z, h
    
        