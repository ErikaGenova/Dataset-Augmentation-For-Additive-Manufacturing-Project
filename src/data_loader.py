# handles dataset & augmentations

import os
import glob
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Transforms for training and validation WITHOUT augmentations
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # TODO: check mean and std
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
}

# Transforms for training and validation WITH augmentations
"""
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
}
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


def get_dataloaders(data_dir, batch_size=16, val_split=0.2, num_workers=4, random_seed=42):
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

    # TODO: maybe it's better to perform cross-validation instead of train/val split

    train_paths = [file_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [file_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_ds = DefectDataset(train_paths, train_labels, transform=data_transforms['train'])
    val_ds   = DefectDataset(val_paths, val_labels, transform=data_transforms['val'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == '__main__':
    import argparse
    # Script to test dataloader sizes and sample batch
    parser = argparse.ArgumentParser(description='Test DataLoader for PBF defects')
    parser.add_argument('--data-dir', type=str, required=True, help='Root folder with Defects/ and NoDefects/')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()

    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        random_seed=args.random_seed
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
