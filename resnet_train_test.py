import argparse

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from trainers.imagenet_resnet import ImageNetResNet

parser = argparse.ArgumentParser(
                    prog = 'Train test MetaFormer')
parser.add_argument('-d', '--data', type=str)
parser.add_argument('-s', '--seed', type=int, default=42)




if __name__ == "__main__":
    args = parser.parse_args()
    dataroot = args.data
    seed = args.seed

    torch.manual_seed(seed)
    print('Initializing Dataset')
    model = ImageNetResNet(data_dir=dataroot, lr=1e-4, num_classes=100)
    print('Dataset initialized')
    trainer = Trainer(
        accelerator='auto',
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        max_epochs=4,
        logger = CSVLogger(save_dir="logs/"),
    )
    print("Training")
    trainer.fit(model)