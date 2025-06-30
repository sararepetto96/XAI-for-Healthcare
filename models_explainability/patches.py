from torch.distributions.normal import Normal
import torch.nn as nn
import torch


def detach_qk(self, q, k):
    
    # Detach the tensors to avoid gradient computation
    q = q.detach()
    k = k.detach()
    
    return q, k

class NonLinearActivation(nn.Module):
    def __init__(self, activation: nn.Module):
        super(NonLinearActivation, self).__init__()
        
        if isinstance(activation,nn.GELU):
            
            print("Using GELU activation")
            #self.activation = lambda x: x*Normal(torch.zeros_like(x),torch.ones_like(x)).cdf(x)
            self.activation = lambda x : activation(x) / x
            
        elif isinstance(activation,nn.Sigmoid):
            self.activation = activation

    def forward(self, x):
        return x*self.activation(x.detach())
    
if __name__ == "__main__":
    x = torch.randn(5,5)
    
    output = nn.Sigmoid()(x)
    print(output)
    
    output = NonLinearActivation(nn.Sigmoid())(x)
    print(output)

