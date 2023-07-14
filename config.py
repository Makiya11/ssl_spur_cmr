
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def configure_loss(target, device):
    # Get class importance weights
    class_sample_count = np.unique(target, return_counts=True)[1]
    weight = 1. / class_sample_count
    class_weights = weight / weight.sum()
    loss_func = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(device))
    return loss_func

def configure_optimizers(opt, model, learning_rate, weight_decay=0.0001):
    if opt=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), 
                                learning_rate,
                                weight_decay=weight_decay)
    elif opt=='adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                learning_rate,
                                weight_decay=weight_decay)
    elif opt=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                learning_rate)
    return optimizer
