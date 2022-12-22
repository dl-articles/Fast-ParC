from timm.models.layers import LayerNorm, DropPath, trunc_normal_, to_2tuple
from torch import nn
import torch
import torch.functional as F

from layers.parc_factory import ParcOperatorVariation, ParCOperator


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


class MetaFormerBlock(nn.Module):
    def __init__(self,
                 dim,
                 drop_path=0.,
                 use_parc = True,
                 layer_scale_init_value=1e-6,
                 variation=ParcOperatorVariation.FAST,
                 mlp_ratio=4.,
                 global_kernel_size=14,
                 use_pe=True):
        super().__init__()
        self.pw = nn.Conv2d(dim, dim, 1)
        self.gcc_1 = nn.Sequential(
            ParCOperator(dim//2, orientation='H', variation=variation,
                         global_kernel_size=global_kernel_size, use_pe=use_pe),
            ParCOperator(dim // 2, orientation='V', variation=variation,
                         global_kernel_size=global_kernel_size, use_pe=use_pe)
        ) if use_parc else nn.Identity()
        self.gcc_2 = nn.Sequential(
            ParCOperator(dim//2, orientation='V', variation=variation,
                         global_kernel_size=global_kernel_size, use_pe=use_pe),
            ParCOperator(dim // 2, orientation='H', variation=variation,
                         global_kernel_size=global_kernel_size, use_pe=use_pe)
        ) if use_parc else nn.Identity()
        self.norm = LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim)
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

def basic_blocks(dim,
                 index,
                 layers,
                 mlp_ratio=4.,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-5,
                 use_parc=True,
                 use_pe=True,
                 variation=ParcOperatorVariation.FAST
                 ):

    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(MetaFormerBlock(
            dim,
            use_parc=use_parc,
            mlp_ratio=mlp_ratio,
            drop_path=block_dpr,
            use_pe=use_pe,
            layer_scale_init_value=layer_scale_init_value,
            variation=variation
            ))
    blocks = nn.Sequential(*blocks)

    return blocks

class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class MetaFormer(nn.Module):
    def __init__(self, layers, embed_dims=None,
                 parcs=None, mlp_ratios=None,
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2,
                 downsamples=None, down_patch_size=3, down_stride=2, down_pad=1,
                 add_pos_embs=None,
                 drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 variation=ParcOperatorVariation.FAST,
                 **kwargs):

        super().__init__()
        print(num_classes)
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=3, embed_dim=embed_dims[0])
        if add_pos_embs is None:
            add_pos_embs = [False] * len(layers)
        if parcs is None:
            parcs = [False] * len(layers)
        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 use_parc=parcs[i],
                                 mlp_ratio=mlp_ratios[i],
                                 drop_path_rate=drop_path_rate,
                                 use_pe =add_pos_embs[i],
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)
        self.norm = LayerNormChannel(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 \
            else nn.Identity()

        self.apply(self.cls_init_weights)

        # init for classification

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        x = self.norm(x)
        # for image classification
        cls_out = self.head(x.mean([-2, -1]))
        return cls_out

