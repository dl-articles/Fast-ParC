import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn.functional import conv2d

class ParC(torch.nn.Module):
    def __init__(self, interpolation_points, in_channels, out_channels,
                 interpolation_type="bilinear", orientation="horizontal", 
                 aggregate_channels = True, image_size = (224, 224)):
        super().__init__()
        self.orientation = orientation
        self.interpolation_points = interpolation_points
        
        self.interpolation_type = interpolation_type
        self.aggregate_channels = aggregate_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

        width_size, height_size = 1, 1
        self.groups = 1
        if not aggregate_channels:
            self.groups = in_channels
            in_channels = 1
        if orientation == "horizontal":
            width_size = interpolation_points
            self.target_width = image_size[1]
        if orientation == "vertical":
            height_size = interpolation_points
            self.target_height = image_size[0]

        self.parameter_tensor = []
        weights = torch.rand((out_channels, in_channels, height_size, width_size))
        weights = torch.nn.parameter.Parameter(weights)
        self.weights = weights

        bias_tensor = torch.rand(out_channels)
        self.bias_parameters = torch.nn.parameter.Parameter(bias_tensor)
    

    def apply_convoution(self, X):
        target_width, target_height = 1, 1
        if self.orientation == "horizontal":
            width_size = self.interpolation_points
            target_width = X.shape[2]
        if self.orientation == "vertical":
            height_size = self.interpolation_points
            target_height = X.shape[3]

        conv_parameters = None
        conv_parameters = F.interpolate(self.weights, size = (target_height, target_width))
        return conv2d(X, weight = conv_parameters, bias = self.bias_parameters, groups = self.groups)
        
    
    def forward(self, X):
        if self.orientation == "vertical":
            X_cat = torch.cat((X, X[:, :, :-1, :]), dim=-2)
        if self.orientation == "horizontal":
            X_cat = torch.cat((X, X[:, :, :, :-1]), dim=-1)
        return self.apply_convoution(X_cat)
