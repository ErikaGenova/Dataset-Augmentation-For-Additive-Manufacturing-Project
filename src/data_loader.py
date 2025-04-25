# handles dataset & augmentations

import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define transforms for training and validation
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

"""
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
"""

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    '''Returns train and validation DataLoader objects.'''
    # Datasets: split images into train/val folders (manually or via code)
    train_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    val_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transforms
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader