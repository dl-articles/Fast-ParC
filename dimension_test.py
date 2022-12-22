import torch
from models.ResParcNet import ParCResNet50
from models.ConvNeXt_parc import ParCConvNeXt
from layers.parc_factory import ParcOperatorVariation

parcresnet = ParCResNet50(100)
parcconvnext = ParCConvNeXt(100, variation=ParcOperatorVariation.BASIC)

pseudo_image = torch.ones([4, 3, 224, 224])
print(parcresnet(pseudo_image).shape)
print(parcconvnext(pseudo_image).shape)
