from torch.distributions.normal import Normal
import torch.nn as nn
import torch


def detach_qk(self, q, k):
    
    # Detach the tensors to avoid gradient computation
    q = q.detach()
    k = k.detach()
    
    return q, k



