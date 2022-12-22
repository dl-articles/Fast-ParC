import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
import torchvision.transforms as transforms


from config.metaformer_config import metaformer_pppa_s12_224
from dataset.ImageNetKaggle import ImageNetKaggle


class ImageNetMetaFormer(LightningModule):
    def __init__(self, data_dir, lr = 1e-4, batch_size=32,
                 num_classes = 500, max_samples=None,):
        super().__init__()
        self.save_hyperparameters()
        self.model = metaformer_pppa_s12_224(num_classes=num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.val_acc = Accuracy()
        self.val_f1 = F1Score()
        self.train_acc = Accuracy()
        self.train_f1 = F1Score()
        # self.lr_factor = lr_factor
        # self.step_tolerance = step_tolerance
        # self.bad_steps = 0
        # self.burnin_steps = burnin_steps
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.max_samples = max_samples
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.data_dir = data_dir
        self.min_loss = float('inf')

    def forward(self, x):
        return self.softmax(self.model(x))

    def training_step(self, batch, batch_nb):
        x, y = batch

        logits = self(x)
        loss = self.loss(logits, y)
        # optim = self.optimizers()
        #
        # if self.step_tolerance and self.global_step > self.burnin_steps:
        #     loss_item = loss.item()
        #     if loss_item < self.min_loss:
        #         self.min_loss = loss_item
        #         self.bad_steps = 0
        #     else:
        #         self.bad_steps += 1
        #
        #     if self.bad_steps > self.step_tolerance:
        #         self.lr = self.lr * self.lr_factor
        #         optim.param_groups[0]['lr'] = self.lr


        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)

        self.log('learning_rate', self.lr, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        return loss

    # def training_epoch_end(self, training_step_outputs):
    #     if self.step_tolerance:
    #         self.min_loss = float('inf')
    #         self.bad_steps = 0

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3,
                                                                  factor=0.5,min_lr=1e-12)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'valid_loss'}
        # return optimizer