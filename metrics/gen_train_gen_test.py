import os
import glob
import torch
import argparse
from PIL import Image
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
# import build_model
from src.model import build_model 

class CvaeSyntheticDataset(Dataset):
    '''Custom dataset reading files and labels from lists.'''
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # TODO: vedi come come sono salvati i dati dalle cvae e fai di conseguenza
        ...

# class GANSyntheticDataset(Dataset): ...

class OriginalDefectDataset(Dataset):
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

"""
# Dataset for the original data
class GEN_Train_DefectDataset(Dataset):
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

# Dataset for the generated images
class GEN_Test_DefectDataset(Dataset):
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


def GEN_train(args):
    # Set up the data type for GPU or CPU
    if args.cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    return


def GEN_test(args):
    # Set up the data type for GPU or CPU
    if args.cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor


    return
"""

def train(train_loader, val_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nTrain samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

    # Build model
    print(f"\nUsing {args.backbone} as backbone")
    model = build_model(backbone=args.backbone, pretrained=args.pretrained)
    model.to(device)
    print(f"\nModel loaded!")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    logs = []
    for epoch in range(args.epochs):
        # Training loop
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation loop
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        logs.append([epoch+1, train_loss, val_loss, train_acc, val_acc])
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint)

    # Save logs to CSV
    df = pd.DataFrame(logs, columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    df.to_csv('logs.csv', index=False)

    print('Training completed')


def test(model, test_loader, cuda):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the best model
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEN train or GEN test")
    parser.add_argument("--mode", type=str, help="GEN_train or GEN_test", required=True)
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--cuda", action='store_true', help='Use CUDA for computation', default=False)

    args = parser.parse_args()

    if args.mode == "GEN_train":
        # Training with the original dataset and test on the synthetic dataset
        
        # Get training and validation data loaders from OriginalDefectDataset

        """
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
        
        # Create datasets and dataloaders
        train_ds = OriginalDefectDataset(train_paths, train_labels, transform=data_transforms['train'])
        val_ds   = OriginalDefectDataset(val_paths, val_labels, transform=data_transforms['val'])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        """
        train_loader = None
        val_loader = None
        train(train_loader, val_loader, args.model, args.cuda)

        test_loader = None # Get test data loader from CvaeSyntheticDataset
        test(args.model, test_loader, args.cuda)


    elif args.mode == "GEN_test": 
        train_loader = None # from SyntheticDataset
        val_loader = None # from SyntheticDataset
        test_loader = None # from OriginalDefectDataset
        # train...
        # test...
    else:
        print("Invalide mode. Choose either 'GEN_train' or 'GEN_test'")
