import argparse

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from trainers.imagenet_metaformer import ImageNetMetaFormer

parser = argparse.ArgumentParser(
                    prog = 'Train MetaFormer')
parser.add_argument('-d', '--data', type=str)
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-c', '--checkpoint', type=str, default='./checkpoint/train.ckpt')
parser.add_argument('project_name', type=str)
parser.add_argument('-e', '--experiment', type=str)



if __name__ == "__main__":
    args = parser.parse_args()
    dataroot = args.data
    seed = args.seed
    checkpoint = args.checkpoint
    experiment = args.experiment
    project = args.project_name

    torch.manual_seed(seed)
    model = ImageNetMetaFormer(data_dir=dataroot, lr=1e-4, num_classes=350, max_samples=1000)
    trainer = Trainer(
        accelerator='auto',
        callbacks=[TQDMProgressBar(refresh_rate=20),
                   ModelCheckpoint(monitor="val_f1", mode="max"),
                   LearningRateMonitor("epoch"),
                   EarlyStopping(monitor='val_loss', patience=15, mode="min", min_delta=0.0000)],
        max_epochs=100,
        logger = WandbLogger(project=project, experiment=experiment),
    )
    trainer.fit(model, ckpt_path=checkpoint)