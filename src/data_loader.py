# handles dataset & augmentations

import os
import glob
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

# Transforms for training and validation WITH and WITHOUT augmentations
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # standard normalization for grayscale
        #transforms.Normalize(mean=[0.5839], std=[0.2074]) # mean and std computed from the dataset
    ]),
    'train_aug': transforms.Compose([ # Augmentations for training
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # standard normalization for grayscale
        #transforms.Normalize(mean=[0.5839], std=[0.2074]) # mean and std computed from the dataset
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # standard normalization for grayscale
        #transforms.Normalize(mean=[0.5839], std=[0.2074]) # mean and std computed from the dataset
    ])
}

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


def get_dataloaders(data_dir, batch_size=16, val_split=0.2, num_workers=4, random_seed=42, test=False, test_split=0.2):
    '''Split Defects/NoDefects into train and val loaders.'''
    classes = ['NoDefects', 'Defects']
    file_paths, labels = [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for ext in ('png', 'jpg', 'jpeg'):
            files = glob.glob(os.path.join(folder, f'*.{ext}'))
            file_paths += files
            labels += [idx] * len(files)

    if test:
        # train/val/test split stratified by label
        train_idx, temp_idx = train_test_split(
            list(range(len(file_paths))),
            test_size=(val_split + test_split),
            stratify=labels,
            random_state=random_seed
        )
        val_size = val_split / (val_split + test_split)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1 - val_size),
            stratify=[labels[i] for i in temp_idx],
            random_state=random_seed
        )

        train_paths = [file_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_paths = [file_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        test_paths = [file_paths[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        test_ds = DefectDataset(test_paths, test_labels, transform=data_transforms['val'])
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
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
        test_loader = None
    
    # Create datasets and dataloaders
    train_ds = DefectDataset(train_paths, train_labels, transform=data_transforms['train'])
    val_ds   = DefectDataset(val_paths, val_labels, transform=data_transforms['val'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_all_data(data_dir):
    '''Load file paths and labels for all images in the dataset.'''
    classes = ['NoDefects', 'Defects']
    file_paths, labels = [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for ext in ('png', 'jpg', 'jpeg'):
            files = glob.glob(os.path.join(folder, f'*.{ext}'))
            file_paths += files
            labels += [idx] * len(files)
    return file_paths, labels


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


def compute_mean_std(data_dir, batch_size=32, num_workers=4):
    '''Compute mean and std of the entire dataset for grayscale images.'''
    file_paths, labels = get_all_data(data_dir)
    dataset = DefectDataset(file_paths, labels, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sum_ = 0.0
    sum_sq = 0.0
    num_pixels = 0

    for imgs, _ in loader:
        B, C, H, W = imgs.shape
        num_pixels += B * H * W
        sum_ += imgs.sum()
        sum_sq += (imgs ** 2).sum()

    mean = sum_ / num_pixels
    var = (sum_sq / num_pixels) - (mean ** 2)
    std = torch.sqrt(var)

    print(f"Dataset mean (grayscale): {mean.item():.4f}")
    print(f"Dataset std (grayscale): {std.item():.4f}")
    return mean.item(), std.item()


if __name__ == '__main__':
    import argparse
    # Script to test dataloader sizes and sample batch
    parser = argparse.ArgumentParser(description='Test DataLoader for PBF defects')
    parser.add_argument('--data-dir', type=str, required=True, help='Root folder with Defects/ and NoDefects/')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--compute-stats', action='store_true', help='Compute mean and std of dataset')
    args = parser.parse_args()

    if args.compute_stats:
        compute_mean_std(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        train_loader, val_loader = get_dataloaders(
            args.data_dir,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            random_seed=args.random_seed
        )

        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
