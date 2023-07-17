import torch
from torch import nn


# different activation functions
class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))


class QuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)


class MultiQuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)**0.5


class LaplacianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.abs(x)/self.a)


class SuperGaussianActivation(nn.Module):
    def __init__(self, a=1., b=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))**self.b


class ExpSinActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.sin(self.a*x))
