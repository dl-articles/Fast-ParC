import torch
from models.ResParcNet import ParCResNet50
from models.ConvNeXt_parc import ParCConvNeXt

parcresnet = ParCResNet50()
parcconvnext = ParCConvNeXt()

pseudo_image = torch.ones([4, 3, 224, 224])
print(parcresnet(pseudo_image).shape)
print(parcconvnext(pseudo_image).shape)
