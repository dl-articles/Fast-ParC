from collections.abc import Iterable

import torch
from torch import nn
from torch.fft import fft, ifft


class FastParCUnit(nn.Module):
    # Interpolation
    # conv_dim->orientation
    # dim->image_size int or tuple
    def __init__(self, channels, dim, conv_dim='V'):
        assert conv_dim in ['V', 'H'], "You can choice only vertical (V) or horizontal (H) dimension"
        super().__init__()
        self.dim = dim
        self.channels = channels
        self.weights = nn.Parameter(torch.rand((channels, dim)), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(channels), requires_grad=True)
        self.conv_dim = conv_dim
        self.conv_dim_idx = -2 if conv_dim == 'V' else -1

    def process_weight_fft(self):
        if self.conv_dim == 'V':
            weights = self.weights.view(1, self.channels, 1, self.dim)
            return torch.conj(fft(weights, dim=self.conv_dim_idx))
        weights = self.weights.view(1, self.channels, self.dim, 1)
        return torch.conj(fft(weights, dim=self.conv_dim_idx))

    def forward(self, x):
        x = fft(x, dim=self.conv_dim_idx)

        x = x * self.process_weight_fft()

        x = ifft(x, dim=self.conv_dim_idx).real

        return x + self.bias.view(1, self.channels, 1, 1)


class FastParC(nn.Module):

    def __init__(self, channels, kernel):
        super(FastParC, self).__init__()
        v = h = kernel
        if type(kernel) is tuple and len(kernel) == 2:
            v, h = kernel
        self.block = nn.Sequential(
            FastParCUnit(channels, v),
            FastParCUnit(channels, h, conv_dim="H")
        )

    def forward(self, x):
        return self.block(x)
