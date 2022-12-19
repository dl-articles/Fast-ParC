from collections.abc import Iterable

import torch
from torch import nn
from torch.fft import fft, ifft


class FastParCUnit(nn.Module):
    def __init__(self, channels, global_kernel_size, orientation='V', use_pe=True):
        assert orientation in ['V', 'H'], "You can choice only vertical (V) or horizontal (H) dimension"
        super().__init__()
        self.channels = channels
        self.global_kernel_size = global_kernel_size
        self.weights = nn.Parameter(torch.rand((channels, global_kernel_size)), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(channels), requires_grad=True)
        self.orientation = orientation
        self.conv_dim_idx = -2 if orientation == 'V' else -1
        if use_pe:
            self.pe = nn.Parameter(torch.randn(self.channels, self.global_kernel_size, 1))

    def process_weight_fft(self, shape):
        weights = self.interpolate(self.weights, shape)
        return torch.conj(fft(weights, dim=self.conv_dim_idx))

    def interpolate(self, x, shape):
        if self.orientation == 'V':
            dim = shape[0]
            reshaped = x.view(1, self.channels,  self.global_kernel_size, 1)
            return nn.functional.interpolate(reshaped, (dim, 1), mode="bilinear")
        dim = shape[1]
        reshaped = x.view(1, self.channels, 1, self.global_kernel_size)
        return nn.functional.interpolate(reshaped, (1, dim), mode="bilinear")

    def add_pos_embedding(self, x, shape):
        if self.pe is None:
            return x

        return x + self.interpolate(self.pe, shape)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.add_pos_embedding(x, (H, W))

        x = fft(x, dim=self.conv_dim_idx)

        x = x * self.process_weight_fft((H, W))

        x = ifft(x, dim=self.conv_dim_idx).real

        return x + self.bias.view(1, self.channels, 1, 1)