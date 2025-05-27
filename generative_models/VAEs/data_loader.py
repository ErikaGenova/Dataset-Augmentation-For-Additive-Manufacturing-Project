# handles dataset & augmentations

import os
import glob
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torchvision
import numpy as np

# To Tensor and Normalize
transform_1 = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # mean and std computed from the dataset
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
}

# To Tensor only
"""
# (28, 28) is the size of the input images for MNIST
transform_2 = torchvision.transforms.Compose([
    torchvision.transforms.Resize((28, 28)),  # Resize images to 28x28
    torchvision.transforms.ToTensor()
])
"""



class DefectDataset(Dataset):
    '''Custom dataset reading files and labels from lists.'''
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(data_dir, batch_size=16, val_split=0.2, num_workers=4, random_seed=42, image_size=512):
    '''Split Defects/NoDefects into train and val loaders.'''
    classes = ['NoDefects', 'Defects']
    file_paths, labels = [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for ext in ('png', 'jpg', 'jpeg'):
            files = glob.glob(os.path.join(folder, f'*.{ext}'))
            file_paths += files
            labels += [idx] * len(files)

    # train/val split stratified by label
    train_idx, val_idx = train_test_split(
        list(range(len(file_paths))),
        test_size=val_split,
        stratify=labels,
        random_state=random_seed
    )

    train_paths = [file_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [file_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),  # Resize images to 512x512
        torchvision.transforms.ToTensor()
        #transforms.Normalize(mean=[0.5], std=[0.5]) # TODO: try to add this 
    ])

    train_ds = DefectDataset(train_paths, train_labels, transform=transform)
    val_ds = DefectDataset(val_paths, val_labels, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def get_dataloader_by_class(file_paths, labels, batch_size=16, val_split=0.2, num_workers=4, random_seed=42, image_size=512):
    # train/val split
    train_idx, val_idx = train_test_split(
        list(range(len(file_paths))),
        test_size=val_split,
        random_state=random_seed
    )

    train_paths = [file_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [file_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Crea i dataset personalizzati
    train_dataset = DefectDataset(train_paths, train_labels, transform=transform)
    test_dataset = DefectDataset(val_paths, val_labels, transform=transform)

    # Crea i dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def get_defect_dataset(data_dir):
    file_paths, labels = [], []
    # return filepaths and labels equal to 1
    for ext in ('png', 'jpg', 'jpeg'):
        files = glob.glob(os.path.join(data_dir, f'*.{ext}'))
        file_paths += files
    labels = np.ones(len(file_paths))
    return file_paths, labels


def get_no_defect_dataset(data_dir):
    file_paths, labels = [], []
    # return filepaths and labels equal to 0
    for ext in ('png', 'jpg', 'jpeg'):
        files = glob.glob(os.path.join(data_dir, f'*.{ext}'))
        file_paths += files
    labels = np.zeros(len(file_paths))
    return file_paths, labels