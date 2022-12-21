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
parser.add_argument('-p', '--checkpoint', type=str)
parser.add_argument('project_name', type=str)
parser.add_argument('-e', '--experiment', type=str)
parser.add_argument('-c', '--classes', type=int, default=100)
parser.add_argument('-n', '--samples', type=int, default=1000)
parser.add_argument('-b', '--batch', type=int, default=32)
parser.add_argument('-l', '--lr', type=float, default=1e-3)


if __name__ == "__main__":
    args = parser.parse_args()
    dataroot = args.data
    seed = args.seed
    checkpoint = args.checkpoint
    experiment = args.experiment
    project = args.project_name
    classes = args.classes
    max_samples = args.samples
    batch = args.batch
    lr = args.lr
    torch.manual_seed(seed)

    logger = WandbLogger(project=project, name=experiment)


    model = ImageNetMetaFormer(data_dir=dataroot, lr=lr, batch_size=batch,
                               num_classes=classes, max_samples=max_samples)

    logger.watch(model)
    trainer = Trainer(
        accelerator='auto',
        callbacks=[TQDMProgressBar(refresh_rate=20),
                   ModelCheckpoint(monitor="valid_f1", mode="max"),
                   LearningRateMonitor("epoch"),
                   EarlyStopping(monitor='valid_loss', patience=15, mode="min", min_delta=0.00)],
        max_epochs=100,
        logger=logger,
    )
    trainer.fit(model, ckpt_path=checkpoint)