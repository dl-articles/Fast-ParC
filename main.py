import torch
from torch import nn

from layers.fast_parc import FastParCUnit


class ConvModel(nn.Module):
    def __init__(self):

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 9, 3),
            nn.Conv2d(9, 18, 3),
            nn.Conv2d(18, 24, 3),
            FastParCUnit(24, 12),
            FastParCUnit(24, 12, orientation='H'),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2904, 256),
            nn.GELU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)

        return self.classifier(out)

import os


import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = ConvModel()

    def forward(self, x):
        return self.l1(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=3,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)