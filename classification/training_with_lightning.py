"""
Example scrips showing training with Pytorch Lightning for a simple 2 class
image classification problem.
"""
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from data import ImageDataset
from classifiers import LitResNet


def train_lightning_model(
        pl_model: pl.LightningModule,
        transform: transforms.Compose,
        root_data_dir: str,
):
    """
    Finetune our model for classification using Pytorch-lightning.

    Args:
        pl_model (LightningModule): A Pytorch Lightning model.
        transform (Compose): A list of transforms to apply to the data.
        root_data_dir (str): location of the data.
    """
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
    trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1)

    # Train the model
    trainer.fit(model, train_loader, val_loader)


def prediction(
        pl_model: pl.LightningModule,
        transform: transforms.Compose,
        prediction_path: str,
        checkpoint_path: str,
):
    """
    Function to predict single values.

    Args:
        pl_model (LightningModule): A Pytorch Lightning model.
        transform (Compose): A list of transforms to apply to the data.
        prediction_path (str): location of the data to be predicted on.
        checkpoint_path (str): location of the saved model weights.
    """
    pl_model = pl_model.load_from_checkpoint(checkpoint_path)

    pl_model.eval()

    image = Image.open(prediction_path).convert("RGB")
    image = transform(image).type(torch.float).unsqueeze(0)

    with torch.no_grad():
        output_pred = pl_model(image)

    probs = torch.softmax(output_pred, dim=1)

    predicted_class = torch.argmax(probs, dim=1).item()

    print(predicted_class)
    

if __name__ == "__main__":
    root_data_dir = "/Users/aloui/python/datasets/dogs_and_cats/"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_lightning_model(
        pl_model=LitResNet,
        transform=transform,
        root_data_dir=root_data_dir,
    )

    checkpoint_path = "lightning_logs/version_6/checkpoints/epoch=0-step=251.ckpt"

    prediction(
        pl_model=LitResNet,
        transform=transform,
        prediction_path=root_data_dir + "single_prediction/Tess.jpg",
        checkpoint_path=checkpoint_path
    )

    prediction(
        pl_model=LitResNet,
        transform=transform,
        prediction_path=root_data_dir + "single_prediction/cat_or_dog_2.jpg",
        checkpoint_path=checkpoint_path
    )
