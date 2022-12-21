import torch

from models.Metaformer_Parc import MetaFormer


def metaformer_pppa_s12_224(**kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    add_pos_embs = [False, False, False, True]
    token_mixers = [True, True, True, True]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    return MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        add_pos_embs=add_pos_embs,
        **kwargs)

model = metaformer_pppa_s12_224()
inp = torch.rand((1, 3, 224, 224))
out = model(inp)