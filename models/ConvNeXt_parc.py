from torch import nn
from layers.parc_factory import ParCOperator, ParcOperatorVariation
from layers.parc_block import ParCBlock

class ParCNextNeck(nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, image_size,
                 init_kernel_size = 14, fast = False, parc_block = False, 
                 variation=ParcOperatorVariation.BASIC):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=image_size)

        self.depthwise_conv = nn.Sequential(nn.Conv2d(input_channels, input_channels,
                                                       kernel_size=7, padding=3, groups=input_channels),
                                             nn.Conv2d(input_channels, input_channels, kernel_size=1))
        if parc_block:
            self.depthwise_conv = ParCBlock(channels=input_channels, 
                                               init_kernel_size=init_kernel_size,
                                               depthwise=False, 
                                               variation=variation)
        
        self.bottleneck_extender = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.bottleneck_reductor = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        self.projector = nn.Identity()
        self.maxpool_if_necessary = nn.Identity()
        if input_channels != out_channels:
            self.projector = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, 
                                       kernel_size=1, stride=2)
            self.maxpool_if_necessary = nn.MaxPool2d(kernel_size=2, stride=2)
            
    
    def forward(self, input):
        x = self.layernorm(input)
        x = self.depthwise_conv(x)
        x = self.bottleneck_extender(x)
        x = nn.GELU()(x)
        x = self.bottleneck_reductor(x)
        x = self.maxpool_if_necessary(x)
        return x + self.projector(input)

class ParCConvNeXt(nn.Module):
    def __init__(self, classes, variation=ParcOperatorVariation.BASIC):
        super().__init__()
        self.initial_layer = nn.Conv2d(3, 48, 4, 4)

        self.first_sequence = nn.Sequential(*([ParCNextNeck(48, 192, 48,
                                              image_size=(56, 56), 
                                              variation=variation) for i in range(2)]+
                                              [ParCNextNeck(48, 192, 96,
                                              image_size=(56, 56), 
                                              variation=variation)]))

        self.second_sequence = nn.Sequential(*([ParCNextNeck(96, 384, 96, 
                                               image_size=(28, 28),
                                               variation=variation) for i in range(2)]+
                                               [ParCNextNeck(96, 384, 192, 
                                               image_size=(28, 28),
                                               variation=variation)]))

        self.third_sequence = nn.Sequential(*([ParCNextNeck(192, 768, 192, 
                                               image_size=(14, 14),
                                               variation=variation) for i in range(6)]+
                                              [ParCNextNeck(192, 768, 192,
                                               image_size=(14, 14), parc_block=True,
                                               variation=variation) for i in range(2)]+
                                              [ParCNextNeck(192, 768, 384, 
                                               image_size=(14, 14), parc_block=True, 
                                               variation=variation)]))

        self.fourth_sequence = nn.Sequential(*([ParCNextNeck(384, 1536, 384, 
                                               image_size=(7, 7), 
                                               variation=variation) for i in range(2)]+
                                               [ParCNextNeck(384, 1536, 384, 
                                               image_size=(7, 7), parc_block=True,
                                               variation=variation)]))
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.linear_classifier = nn.Linear(384, classes)
    
    def forward(self, input):
        x = self.initial_layer(input)
        x = self.first_sequence(x)
        x = self.second_sequence(x)
        x = self.third_sequence(x)
        x = self.fourth_sequence(x)
        x = self.avg_pool(x)
        x = x.squeeze()
        x = self.linear_classifier(x)
        return x
