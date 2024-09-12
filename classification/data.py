"""
Stores the custom pytorch dataset class.
"""
import os
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """Finetune classification dataset."""

    def __init__(self, root: str, transform: Optional[Callable]=None) -> None:
        """
        Arguments:
            root (string): Directory with all the images.
            transform (Callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform

        file_paths = []
        labels = []

        classes = sorted(os.listdir(root))
        for label_idx, class_name in enumerate(classes):
            class_folder = os.path.join(root, class_name)

            for file_name in os.listdir(class_folder):
                if file_name.endswith('.jpg'):
                    file_paths.append(os.path.join(class_folder, file_name))
                    labels.append(label_idx)

        self.file_paths_array = np.array(file_paths)
        self.labels_array = np.array(labels)
        self.class_mapping = {class_name: index for index, class_name in
                              enumerate(classes)}

    def __len__(self) -> int:
        return len(self.file_paths_array)
    
    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        label = torch.tensor(self.labels_array[idx], dtype=torch.long)
        image = Image.open(self.file_paths_array[idx]).convert("RGB")

        if self.transform:
            return self.transform(image).type(torch.float), label

        return transforms.ToTensor()(image).type(torch.float), label


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(
        # root="/mnt/c/Users/aloui/python/datasets/dogs_and_cats/test_set",
        root="/Users/aloui/python/datasets/dogs_and_cats/test_set",
        transform=transform
        )

    print("y labels: ", dataset.labels_array)
    print("X paths: ", dataset.file_paths_array)
    print("Classes: ", dataset.class_mapping)
    print("Data length: ", dataset.__len__)
    print("Example X: ", dataset.__getitem__(0))
    print("Example X shape: ", dataset.__getitem__(0)[0].shape)
