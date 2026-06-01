import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ContrastiveLearningViewGenerator(object):
    """Generates two views of the same image for SimCLR."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

def get_simclr_pipeline(img_size=32):
    """Returns specialized data augmentations for contrastive learning."""
    color_jitter = transforms.ColorJitter(0.8*1.0, 0.8*1.0, 0.8*1.0, 0.2*1.0)
    return transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

def get_mae_pipeline(img_size=32):
    """Standard normalization pipeline for MAE reconstruction."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

class UnlabeledDataset(Dataset):
    """Wrapper dataset for handling raw unannotated image matrices/arrays."""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        # Handle numpy arrays or tensors appropriately
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if self.transform:
            return self.transform(img)
        return img