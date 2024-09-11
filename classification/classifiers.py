"""
Pytorch and pytorch-lightning classifiers.
"""
from torch import nn, optim
from torchvision.models import resnet50
import pytorch_lightning as pl


class LitResNet(pl.LightningModule):
    """
    Pytorch-lightning ResNet50 model for finetuneing for classification
    problems.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)