from collections.abc import Iterable

import torch
from torch import nn
from torch.fft import fft, ifft


class FastParCUnit(nn.Module):
    # Interpolation
    # conv_dim->orientation
    # dim->image_size int or tuple
    def __init__(self, channels, global_kernel_size, image_size, depthwise=False, orientation='V'):
        assert orientation in ['V', 'H'], "You can choice only vertical (V) or horizontal (H) dimension"
        super().__init__()
        self.image_height = self.image_width = image_size
        if type(image_size) is tuple and len(image_size) == 2:
            self.image_height, self.image_width = image_size
        self.channels = channels
        self.depthwise = depthwise
        self.weights = nn.Parameter(torch.rand((channels, global_kernel_size)), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(channels), requires_grad=True)
        self.orientation = orientation
        self.conv_dim_idx = -2 if orientation == 'V' else -1

    def process_weight_fft(self):
        if self.orientation == 'V':
            weights = self.weights.view(1, self.channels, 1, self.kernel)
            weights = nn.functional.interpolate(weights, (1, self.image_width), mode="bilinear")
        else:
            weights = self.weights.view(1, self.channels, self.kernel, 1)
            weights = nn.functional.interpolate(weights, (self.image_height, 1), mode="bilinear")
        return torch.conj(fft(weights, dim=self.conv_dim_idx))

    def forward(self, x):
        x = fft(x, dim=self.conv_dim_idx)

        x = x * self.process_weight_fft()

        x = ifft(x, dim=self.conv_dim_idx).real

        return x + self.bias.view(1, self.channels, 1, 1)