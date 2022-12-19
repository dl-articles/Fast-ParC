import argparse

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from trainers.imagenet_metaformer import ImageNetMetaFormer

parser = argparse.ArgumentParser(
                    prog = 'Sweep MetaFormer')
parser.add_argument('project_name', type=str)
parser.add_argument('experiment_name', type=str)
parser.add_argument('-d', type=str)
parser.add_argument('-s', type=int, default=42)

sweep_config = {
  "method": "random",   # Random search
  "metric": {           # We want to maximize val_acc
      "name": "valid_acc",
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
    experiment = args['experiment_name']
    project = args['project_name']
    dataroot = args['-d']
    seed = args['-s']

    torch.manual_seed(seed)


    sweep_id = wandb.sweep(sweep_config, project=project)
    run = wandb.init(name=experiment)
    def sweep_iter():

        wandb_logger = WandbLogger()

        model = ImageNetMetaFormer(data_dir=dataroot, lr=wandb.config.lr)
        trainer = Trainer(
            logger=wandb_logger,  # W&B integration
            gpus=-1,  # use all GPU's
            max_epochs=4  # number of epochs
        )
        trainer.fit(model)


    wandb.agent(sweep_id, function=sweep_iter)
    run.finish()