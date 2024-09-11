"""
Example scrips showing training with Pytorch Lightning for a simple 2 class
image classification problem.
"""
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from data import ImageDataset
from classifiers import LitResNet

def train_lightning_model(pl_model: pl.LightningModule):
    """
    Finetune our model for classification using Pytorch-lightning.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_data_dir = "/Users/aloui/python/datasets/dogs_and_cats/"
    train_dataset = ImageDataset(
        root=root_data_dir + "training_set",
        transform=transform
        )
    val_dataset = ImageDataset(
        root=root_data_dir + "test_set",
        transform=transform
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=15
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=15
        )

    # Initialise the model
    model = pl_model(num_classes=2)

    # Initialise the trainer
    trainer = pl.Trainer(max_epochs=1)

    # Train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train_lightning_model(pl_model=LitResNet)
