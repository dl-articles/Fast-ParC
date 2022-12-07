import torch
from models.ResParcNet import ParCResNet50

parcresnet = ParCResNet50()

pseudo_image = torch.ones([4, 3, 224, 224])
print(parcresnet(pseudo_image).shape)