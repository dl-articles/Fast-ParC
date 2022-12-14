from torch import nn
from layers.parc_block import ParCBlock

class ParCNeck(nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, image_size,
                 init_kernel_size = 10, fast = False, seq_transition=False, project=False):
        super().__init__()
        self.downsampler = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        if not fast:
            self.transitioner = ParCBlock(hidden_channels, init_kernel_size, 
                                        image_size, depthwise=True)
        self.upsampler = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(hidden_channels)
        self.shortcut = nn.Identity()
        self.maxpool_if_necessary = nn.Identity()
        if project:
            self.shortcut = nn.Conv2d(input_channels, out_channels, kernel_size=1)
        if seq_transition:
            self.maxpool_if_necessary = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, input):
        x = self.downsampler(input)
        x = self.transitioner(x)
        x = nn.ReLU()(self.batchnorm(x))
        x = self.upsampler(x)
        x = self.maxpool_if_necessary(x)
        return self.shortcut(input)+x
 
class ParCResNet50(nn.Module):
    def __init__(self, classes) -> None:
        super().__init__()
        
        self.initial_layer = nn.Conv2d(kernel_size=7, in_channels=3, out_channels=64, 
                                       stride=2, padding=3)
        self.initial_maxpool_layer = nn.Conv2d(kernel_size=3, stride=2, 
                                               in_channels=64, out_channels=64, padding=1)

        self.first_seq = nn.Sequential(*([ParCNeck(64, 64, 256, 
                                            image_size=(56, 56), project=True),
                                          ParCNeck(256, 64, 256, 
                                            image_size=(56, 56)),
                                          ParCNeck(256, 64, 128, 
                                            image_size=(56, 56), seq_transition=True)]))

        self.second_seq = nn.Sequential(*([ParCNeck(128, 128, 512, 
                                            image_size=(28, 28), project=True)]+
                                          [ParCNeck(512, 128, 512, 
                                            image_size=(28, 28)) for i in range(2)]+
                                          [ParCNeck(512, 128, 256, 
                                            image_size=(28, 28), seq_transition=True)]))

        self.third_seq = nn.Sequential(*([ParCNeck(256, 256, 1024, 
                                            image_size=(14, 14), project=True)]+
                                         [ParCNeck(1024, 256, 1024, 
                                            image_size=(14, 14)) for i in range(4)]+
                                         [ParCNeck(1024, 256, 512, 
                                            image_size=(14, 14), seq_transition=True)]))

        self.fourth_seq = nn.Sequential(*([ParCNeck(512, 512, 2048, 
                                            image_size=(7, 7), project=True)]+
                                          [ParCNeck(2048, 512, 2048, 
                                            image_size=(7, 7)) for i in range(2)]))
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.linear_classifier = nn.Linear(2048, classes)

    def forward(self, input):
        x = self.initial_maxpool_layer(self.initial_layer(input))
        x = self.first_seq(x)
        x = self.second_seq(x)
        x = self.third_seq(x)
        x = self.fourth_seq(x)
        x = self.avg_pool(x)
        x = x.squeeze()
        x = self.linear_classifier(x)
        return x
