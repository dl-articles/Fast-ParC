import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d
from torch import nn

class ParCUnit(torch.nn.Module):
    # Image_height, image_width -> size (int or tuple)
    # In_channels==out_channels
    # Fast Parc won't work with in_channels != out_channels
    # And it will be able to perform only depthwise convolution
    def __init__(self, channels, init_kernel_size, use_pe = True,
                orientation="H", depthwise = False): # Depthwise -> True
        super().__init__()

        self.orientation = orientation

        self.init_kernel_size = init_kernel_size
        self.channels = channels
        in_channels, out_channels = channels, channels

        width_size, height_size = 1, 1
        self.groups = 1
        if depthwise:
            self.groups = in_channels
            in_channels = 1
        if orientation == "H":
            width_size = init_kernel_size
        if orientation == "V":
            height_size = init_kernel_size

        weights = torch.rand((out_channels, in_channels, height_size, width_size))
        weights = torch.nn.parameter.Parameter(weights)
        self.weights = weights

        bias_tensor = torch.rand(out_channels)
        self.bias_parameters = torch.nn.parameter.Parameter(bias_tensor)
        if use_pe:
            self.pe = nn.Parameter(torch.randn(channels, self.init_kernel_size, 1))

    def apply_convoution(self, X):
        target_width, target_height = 1, 1
        if self.orientation == "H":
            target_width = X.shape[2]
        if self.orientation == "V":
            target_height = X.shape[3]

        conv_parameters = None
        conv_parameters = F.interpolate(self.weights, size = (target_height, target_width),
                                        mode = "bilinear")
        
        output = conv2d(X, weight = conv_parameters, bias = self.bias_parameters, groups = self.groups)
        return output

    def interpolate(self, x, shape):
        if self.orientation == 'V':
            dim = shape[0]
            reshaped = x.view(1, self.channels,  self.init_kernel_size, 1)
            return nn.functional.interpolate(reshaped, (dim, 1), mode="bilinear")
        dim = shape[1]
        reshaped = x.view(1, self.channels, 1, self.init_kernel_size)
        return nn.functional.interpolate(reshaped, (1, dim), mode="bilinear")    
        
    def add_pos_embedding(self, x, shape):
        if self.pe is None:
            return x
        return x + self.interpolate(self.pe, shape)
    
    def forward(self, X):
        *_, H, W = X.shape
        X = self.add_pos_embedding(X, (H, W))
        if self.orientation == "H":
            X_cat = torch.cat((X, X[:, :, :, :-1]), dim=-1)
        if self.orientation == "V":
            X_cat = torch.cat((X, X[:, :, :-1, :]), dim=-2)
        return self.apply_convoution(X_cat)
