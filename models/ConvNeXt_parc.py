from torch import nn
from layers.parc import ParCBlock

class ParCNextNeck(nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, image_size,
                 init_kernel_size = 10, fast = False):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=image_size)
        if not fast:
            self.parc_block = ParCBlock(input_channels, init_kernel_size, 
                                        image_size, depthwise=True)
        
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
        x = self.parc_block(x)
        x = self.bottleneck_extender(x)
        x = nn.GELU()(x)
        x = self.bottleneck_reductor(x)
        x = self.maxpool_if_necessary(x)
        return x + self.projector(input)

class ParCConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_layer = nn.Conv2d(3, 48, 4, 4)

        self.first_sequence = nn.Sequential(*([ParCNextNeck(48, 192, 48,
                                              image_size=(56, 56)) for i in range(2)]+
                                              [ParCNextNeck(48, 192, 96,
                                              image_size=(56, 56))]))

        self.second_sequence = nn.Sequential(*([ParCNextNeck(96, 384, 96, 
                                               image_size=(28, 28)) for i in range(2)]+
                                               [ParCNextNeck(96, 384, 192, 
                                               image_size=(28, 28))]))

        self.third_sequence = nn.Sequential(*([ParCNextNeck(192, 768, 192, 
                                               image_size=(14, 14)) for i in range(8)]+
                                              [ParCNextNeck(192, 768, 384, 
                                               image_size=(14, 14))]))

        self.fourth_sequence = nn.Sequential(*([ParCNextNeck(384, 1536, 384, 
                                               image_size=(7, 7)) for i in range(3)]))
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.linear_classifier = nn.Linear(384, 1000)
    
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
