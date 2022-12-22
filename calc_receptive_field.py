import argparse

import numpy as np
import torch
from receptivefield.pytorch import PytorchReceptiveField
from torch import nn

from config.metaformer_config import metaformer_pppa_s12_224
from dataset.ImageNetKaggle import ImageNetKaggle

parser = argparse.ArgumentParser(
                    prog = 'Calculate receptive field')
parser.add_argument('-d', '--data', type=str)
parser.add_argument('-s', '--seed', type=int, default=42)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    root = args.data
    seed = args.seed

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = ImageNetKaggle(root, split='val')

    rand_image_idx = np.random.randint(0, len(dataset)-1)

    image, _ = dataset[rand_image_idx]

    def metaformer():
        model = metaformer_pppa_s12_224()
        model.head = nn.Identity()
        model.eval()
        return model.to(device)

    rf = PytorchReceptiveField(metaformer)
    rf.compute((224, 224, 3))

    rf.plot_gradient_at(fm_id=0, point=(8, 8),image=image)


