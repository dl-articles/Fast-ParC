import argparse

import torch
import torchvision
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from trainers.imagenet_models import ImageNetModel

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
parser.add_argument('-m', '--model', type=str)
parser.add_argument('-t', '--step_tolerance', type=int)
parser.add_argument('-f', '--burnin', type=int)



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
    model_name = args.model
    lr = args.lr
    step_tolerance = args.step_tolerance
    burnin = args.burnin
    torch.manual_seed(seed)

    logger = WandbLogger(project=project, name=experiment, entity="dl-artcls")

    model = ImageNetModel(data_dir=dataroot, model_name=model_name, lr=lr,
                          batch_size=batch, burnin_steps=burnin, step_tolerance=step_tolerance,
                          num_classes=classes, max_samples=max_samples)

    logger.watch(model)
    trainer = Trainer(
        accelerator='auto',
        callbacks=[TQDMProgressBar(refresh_rate=20),
                   ModelCheckpoint(monitor="valid_f1", mode="max"),
                   LearningRateMonitor("step")],
        max_epochs=100,
        logger=logger,
    )
    trainer.fit(model, ckpt_path=checkpoint)
