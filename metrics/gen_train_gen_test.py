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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEN train or GEN test")
    parser.add_argument("--mode", type=str, help="GEN_train or GEN_test", required=True)
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--cuda", action='store_true', help='Use CUDA for computation', default=False)

    args = parser.parse_args()

    if args.mode == "GEN_train":
        GEN_train(args)
    elif args.mode == "GEN_test":
        GEN_test(args)
    else:
        print("Invalide mode. Choose either 'GEN_train' or 'GEN_test'")
