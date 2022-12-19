import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import torchvision.transforms as transforms


from config.metaformer_config import metaformer_pppa_s12_224
from dataset.ImageNetKaggle import ImageNetKaggle


class ImageNetMetaFormer(LightningModule):
    def __init__(self, data_dir, lr = 1e-4, batch_size=32):
        super().__init__()
        self.model = metaformer_pppa_s12_224()
        self.softmax = nn.Softmax(dim=1)
        self.val_acc = Accuracy()
        self.batch_size = batch_size
        self.loss = nn.NLLLoss()
        self.lr = lr
        self.data_dir = data_dir

    def forward(self, x):
        return self.softmax(self.model(x))

    def training_step(self, batch, batch_nb):
        x, y = batch

        loss = self.loss(self.model(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)

        # Log validation loss (will be automatically averaged over an epoch)
        self.log('valid_loss', loss)
        self.log('valid_acc', self.val_acc)

    def setup(self, stage=None):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        self.imagenet_train = ImageNetKaggle(self.data_dir, split='train', transform=train_transform)
        self.imagenet_val = ImageNetKaggle(self.data_dir, split='val', transform=val_transform)

    def train_dataloader(self):
        return DataLoader(self.imagenet_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.imagenet_val, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)