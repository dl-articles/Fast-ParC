import torch
from layers.parc_factory import ParCOperator, ParcOperatorVariation
from torch import nn

class ParCBlock(nn.Module):
    def __init__(self, channels, init_kernel_size, variation, use_pe = True,
                depthwise = False):
        super().__init__()
                
        self.parc_h = ParCOperator(channels=channels//2, init_kernel_size=init_kernel_size, 
                                    use_pe = use_pe, orientation="H", 
                                    depthwise = depthwise, variation=variation)
                
        self.parc_v = ParCOperator(channels=channels//2, init_kernel_size=init_kernel_size, 
                                    use_pe = use_pe, orientation="V", 
                                    depthwise = depthwise, variation=variation)
        
    def forward(self, input):
        channels = input.shape[1]
        input_h = input[:, :channels//2, :, :]
        input_v = input[:, channels//2:, :, :]
        output_h = self.parc_h(input_h)
        output_v = self.parc_v(input_v)
        return torch.cat((output_h, output_v), dim=1)
