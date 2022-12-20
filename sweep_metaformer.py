import argparse

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from trainers.imagenet_metaformer import ImageNetMetaFormer

parser = argparse.ArgumentParser(
                    prog = 'Sweep MetaFormer')
parser.add_argument('project_name', type=str)
parser.add_argument('-e', '--experiment', type=str)
parser.add_argument('-d', '--data', type=str)
parser.add_argument('-s', '--seed', type=int, default=42)

sweep_config = {
  "method": "random",   # Random search
  "metric": {           # We want to maximize val_acc
      "name": "val_acc",
      "goal": "maximize"
  },
  "parameters": {
        "lr": {
            # log uniform distribution between exp(min) and exp(max)
            "distribution": "log_uniform",
            "min": -11.512,   # exp(-9.21) = 1e-4
            "max": -4.61    # exp(-4.61) = 1e-2
        }
    }
}





if __name__ == "__main__":
    args = parser.parse_args()
    experiment = args.experiment
    project = args.project_name
    dataroot = args.data
    seed = args.seed

    torch.manual_seed(seed)


    sweep_id = wandb.sweep(sweep_config, project=project)
    def sweep_iter():
        wandb.init(name=experiment)
        wandb_logger = WandbLogger()

        model = ImageNetMetaFormer(data_dir=dataroot, lr=wandb.config.lr)
        trainer = Trainer(
            logger=wandb_logger,  # W&B integration
            accelerator='auto',
            max_epochs=4  # number of epochs
        )
        trainer.fit(model)
        wandb.finish()


    wandb.agent(sweep_id, function=sweep_iter)