"""
Implementations of SimCLR in PyTorch

A Simple Framework for Contrastive Learning of Visual Representations:
    https://arxiv.org/pdf/2002.05709.pdf
Acknowledgments:
    https://github.com/Lightning-AI/lightning-bolts
    https://github.com/google-research/simclr
"""
import math
from argparse import ArgumentParser

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Projection(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)

class SimCLR(nn.Module):
    def __init__(self, encoder, hidden_dim=512, out_dim=128,
                temperature=0.1, learning_rate=1e-5, weight_decay=1e-6):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            lr: the optimizer learning rate
        """
        
        super().__init__()

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.weight_decay = weight_decay
        self.temperature = temperature
        self.learning_rate = learning_rate
        
        self.projection = Projection(input_dim=self.hidden_dim, 
                                     hidden_dim=self.hidden_dim, 
                                     output_dim=self.out_dim)
        self.optimizer = self.configure_optimizers()
    
    def forward(self, x1, x2):

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2)
        return loss


    def nt_xent_loss(self, out_1, out_2, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        emb_i =  out_1
        emb_j = out_2

        ## normalize embeddings
        z_i = F.normalize(emb_i.flatten(1), dim=1) #n_batch x n_emb
        z_j = F.normalize(emb_j.flatten(1), dim=1) #n_batch x n_emb

        r = torch.cat([z_i, z_j], dim=0) #2*n_batch x n_emb
        r_dist = torch.cat([z_i, z_j], dim=0)
        cov = torch.mm(r, r_dist.t().contiguous()) #n_batch x n_batch
        sim = torch.exp(cov/self.temperature)
        neg = sim.sum(dim=-1) #n_batch

        ### now for each row, subtract the true positives
        row_sub = Tensor(neg.shape).fill_(math.e ** (1/self.temperature)).to(neg.device)
        neg = torch.clamp(neg-row_sub, min=eps)

        ### get the true positives
        pos = torch.exp(torch.sum(z_i*z_j, dim=-1)/self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos/(neg + eps)).mean()
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.projection.parameters()), 
                                    lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        return optimizer

