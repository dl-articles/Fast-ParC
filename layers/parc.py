import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d
from torch import nn

class ParCUnit(torch.nn.Module):
    def __init__(self, interpolation_points, in_channels, out_channels,
                 image_height, image_width,
                 interpolation_type="bilinear", orientation="H", 
                 aggregate_channels = True):
        super().__init__()

        self.orientation = orientation
        self.interpolation_type = interpolation_type

        width_size, height_size = 1, 1
        self.groups = 1
        if not aggregate_channels:
            self.groups = in_channels
            in_channels = 1
        if orientation == "H":
            width_size = interpolation_points
            image_height = 1
        if orientation == "V":
            height_size = interpolation_points
            image_width = 1
        positional_codes_values = torch.rand(image_height, image_width)

        weights = torch.rand((out_channels, in_channels, height_size, width_size))
        weights = torch.nn.parameter.Parameter(weights)
        self.weights = weights

        bias_tensor = torch.rand(out_channels)
        self.bias_parameters = torch.nn.parameter.Parameter(bias_tensor)

        self.positional_encoding = torch.nn.parameter.Parameter(positional_codes_values)

    def apply_convoution(self, X):
        target_width, target_height = 1, 1
        if self.orientation == "H":
            target_width = X.shape[2]
        if self.orientation == "V":
            target_height = X.shape[3]

        conv_parameters = None
        conv_parameters = F.interpolate(self.weights, size = (target_height, target_width),
                                        mode = self.interpolation_type)
        
        output = conv2d(X, weight = conv_parameters, bias = self.bias_parameters, groups = self.groups)
        output += self.positional_encoding
        return output
        
    
    def forward(self, X):
        if self.orientation == "H":
            X_cat = torch.cat((X, X[:, :, :, :-1]), dim=-1)
        if self.orientation == "V":
            X_cat = torch.cat((X, X[:, :, :-1, :]), dim=-2)
        return self.apply_convoution(X_cat)


class ParCBlock(nn.Module):
    def __init__(self, interpolation_points, in_channels, out_channels,
                 image_height, image_width,
                 interpolation_type = "bilinear", aggregate_channels = True):
        super().__init__()
                
        self.parc_h = ParCUnit(interpolation_points=interpolation_points, orientation="H",
            in_channels=in_channels//2, out_channels=out_channels//2, 
            image_height=image_height, image_width=image_width,
            interpolation_type=interpolation_type, aggregate_channels=aggregate_channels)
                
        self.parc_v = ParCUnit(interpolation_points=interpolation_points, orientation="V",
            in_channels=in_channels//2, out_channels=out_channels//2, 
            image_height=image_height, image_width=image_width,
            interpolation_type=interpolation_type, aggregate_channels=aggregate_channels)
        
        
    def forward(self, input):
        channels = input.shape[1]
        input_h = input[:, :channels//2, :, :]
        input_v = input[:, channels//2:, :, :]
        output_h = self.parc_h(input_h)
        output_v = self.parc_v(input_v)
        return torch.cat((output_h, output_v), dim=1)