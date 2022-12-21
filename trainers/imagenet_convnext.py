import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
import torchvision.transforms as transforms


from config.metaformer_config import metaformer_pppa_s12_224
from models.ConvNeXt_parc import ParCConvNeXt
from dataset.ImageNetKaggle import ImageNetKaggle


class ImageNetConvNext(LightningModule):
    def __init__(self, data_dir, lr = 1e-4, batch_size=32,
                 num_classes = 500, max_samples=None, weight_decay = 1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.model = ParCConvNeXt()
        self.softmax = nn.Softmax(dim=1)
        self.val_acc = Accuracy()
        self.val_f1 = F1Score()
        self.train_acc = Accuracy()
        self.train_f1 = F1Score()
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.max_samples = max_samples
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.data_dir = data_dir

    def forward(self, x):
        return self.softmax(self.model(x))

    def training_step(self, batch, batch_nb):
        x, y = batch

        logits = self(x)
        loss = self.loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)

        self.log('valid_loss', loss)
        self.log('valid_acc', self.val_acc, prog_bar=True)
        self.log('valid_f1', self.val_f1, prog_bar=True)
        return loss

    def setup(self, stage=None):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        self.imagenet_train = ImageNetKaggle(self.data_dir, split='train',
                                             restrict_classes=self.num_classes,
                                             max_samples = self.max_samples,
                                             transform=train_transform)
        self.imagenet_val = ImageNetKaggle(self.data_dir, split='val',
                                           restrict_classes=self.num_classes,transform=val_transform)

    def train_dataloader(self):
        return DataLoader(self.imagenet_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.imagenet_val, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=10, epochs=4)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler,}