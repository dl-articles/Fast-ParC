from timm.models import DropPath, trunc_normal_
from timm.models.layers import LayerNorm
from torch import nn
import torch
import torch.functional as F

from layers.parc_factory import ParcOperatorVariation, ParCOperator


class SqEx(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MetaformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 variation=ParcOperatorVariation.FAST,
                 pw_ratio=4.,
                 global_kernel_size=14,
                 use_pe=True):
        super().__init__()
        self.pw = nn.Conv2d(dim, dim, 1)
        self.gcc_1 = nn.Sequential(
            ParCOperator(dim//2, orientation='H', variation=variation,
                         global_kernel_size=global_kernel_size, use_pe=use_pe),
            ParCOperator(dim // 2, orientation='V', variation=variation,
                         global_kernel_size=global_kernel_size, use_pe=use_pe)
        )
        self.gcc_2  = nn.Sequential(
            ParCOperator(dim//2, orientation='H', variation=variation,
                         global_kernel_size=global_kernel_size, use_pe=use_pe),
            ParCOperator(dim // 2, orientation='V', variation=variation,
                         global_kernel_size=global_kernel_size, use_pe=use_pe)
        )
        self.norm = LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, pw_ratio * dim),
            nn.GELU(),
            nn.Linear(pw_ratio * dim, dim)
        )
        self.channels_attention = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                               requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        input = x
        x = self.pw(x)
        x_H, x_W = torch.chunk(x, 2, dim=1)
        x_H, x_W = self.gcc_1(x_H), self.gcc_2(x_W)
        x = torch.cat((x_H, x_W), dim=1)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.mlp(x)
        if self.channels_attention is not None:
            x = self.channels_attention * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x

class MetaFormer(nn.Module):
    def __init__(self):
        super().__init__()

