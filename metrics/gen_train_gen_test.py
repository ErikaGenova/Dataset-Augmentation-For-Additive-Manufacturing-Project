import os
import re
import glob
import sys
import torch
import argparse
import pandas as pd
import torch.utils.data
import torch.optim as optim
import torchvision.transforms as transforms

from torch import nn
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import build_model
from src.model import build_model 

# Dataset for the synthetic data by VAE
class VAESyntheticDataset(Dataset):
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

# Dataset for the synthetic data by CVAE
class CvaeSyntheticDataset(Dataset):
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

# Dataset for the synthetic data by GANs
class GANSyntheticDataset(Dataset):
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

# Dataset for the synthetic data by Diffuson Models
class DiffusionSyntheticDataset(Dataset):
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

# Dataset for the synthetic data by SinGAN
class SinGANSyntheticDataset(Dataset):
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

# Dataset for the original data
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

# get all paths and labels of original dataset
def get_original_images(args):
    # path of the original dataset
    data_dir = os.path.join(args.base_dir, "images", "original")

    classes = ['NoDefects', 'Defects']
    file_paths, labels = [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for ext in ('png', 'jpg', 'jpeg'):
            files = glob.glob(os.path.join(folder, f'*.{ext}'))
            file_paths += files
            labels += [idx] * len(files)

    return file_paths, labels

# get all paths and labels of generated images by CVAE
def get_generated_images(args):
    file_paths, labels = [], []

    for ext in ('png', 'jpg', 'jpeg'):
        files = glob.glob(os.path.join(args.dir_generated, f'*.{ext}'))
        for path in files:
            filename = os.path.basename(path)
            match = re.search(r'class_(\d+)', filename)
            if match:
                class_id = int(match.group(1))
                file_paths.append(path)
                labels.append(class_id)

    return file_paths, labels

# get transformations to have size 
def get_transformations(args):
    # Estensioni dei file immagine da considerare
    image_extensions = ('*.png', '*.jpg', '*.jpeg')

    # Lista di tutti i file immagine
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(args.dir_generated, ext)))

    # Verifica se sono presenti immagini
    if not all_images:
        raise FileNotFoundError(f"No images found in {args.dir_generated}")

    # Prendi il primo file immagine
    first_image_path = all_images[0]

    # Ottieni le dimensioni (larghezza, altezza) dell'immagine
    with Image.open(first_image_path) as img:
        original_size = img.size  # (width, height)

    # Transforms for training and validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(original_size[::-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # standard normalization for grayscale
        ]),
        'val': transforms.Compose([
            transforms.Resize(original_size[::-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # standard normalization for grayscale
        ])
    }

    return data_transforms

# get train and val dataloaders
def get_train_val_dataloaders(args):
    train_loader = None
    val_loader = None

    # parameters
    batch_size = args.batch_size
    val_split = args.val_split
    num_workers = args.num_workers
    random_seed = args.random_seed

    # In this case, the training and validation dataloaders are created from the original dataset
    if args.mode == "GEN_train":
        
        file_paths, labels = get_original_images(args)

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
        train_ds = OriginalDefectDataset(train_paths, train_labels, transform=args.data_transforms['train'])
        val_ds   = OriginalDefectDataset(val_paths, val_labels, transform=args.data_transforms['val'])
    
    # In this case, the training and validation dataloaders are created from the generated dataset based on the model and experiment
    else:

        # get vectors of paths and labels of VAE's images
        file_paths, labels = get_generated_images(args)

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

        if args.model == "VAE":
            # Create datasets and dataloaders
            train_ds = VAESyntheticDataset(train_paths, train_labels, transform=args.data_transforms['train'])
            val_ds   = VAESyntheticDataset(val_paths, val_labels, transform=args.data_transforms['val'])
        
        if args.model == "CVAE":
            # Create datasets and dataloaders
            train_ds = CvaeSyntheticDataset(train_paths, train_labels, transform=args.data_transforms['train'])
            val_ds   = CvaeSyntheticDataset(val_paths, val_labels, transform=args.data_transforms['val'])
        
        if args.model == "GANs":
            # Create datasets and dataloaders
            train_ds = GANSyntheticDataset(train_paths, train_labels, transform=args.data_transforms['train'])
            val_ds   = GANSyntheticDataset(val_paths, val_labels, transform=args.data_transforms['val'])
        
        if args.model == "Diffusion":
            # Create datasets and dataloaders
            train_ds = DiffusionSyntheticDataset(train_paths, train_labels, transform=args.data_transforms['train'])
            val_ds   = DiffusionSyntheticDataset(val_paths, val_labels, transform=args.data_transforms['val'])
        
        if args.model == "SinGAN":
            # Create datasets and dataloaders
            train_ds = SinGANSyntheticDataset(train_paths, train_labels, transform=args.data_transforms['train'])
            val_ds   = SinGANSyntheticDataset(val_paths, val_labels, transform=args.data_transforms['val'])

        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

# get only test dataloader
def get_test_dataloader(args):
    test_loader = None

    # parameters
    batch_size = args.batch_size
    num_workers = args.num_workers

    # in this case the train dataloader is created from genereted dataset based on args.model
    if args.mode == "GEN_train":

        # get vectors of paths and labels of VAE's images
        test_paths, test_labels = get_generated_images(args)

        # test dataloader with VAE
        if args.model == "VAE":
            test_ds = VAESyntheticDataset(test_paths, test_labels, transform=args.data_transforms['val'])

        # test dataloader with CVAE
        if args.model == "CVAE":
            test_ds = CvaeSyntheticDataset(test_paths, test_labels, transform=args.data_transforms['val'])
        
        # test dataloader with GANs
        if args.model == "GANs":
            test_ds = CvaeSyntheticDataset(test_paths, test_labels, transform=args.data_transforms['val'])

        # test dataloader with Diffusion
        if args.model == "Diffusion":
            test_ds = DiffusionSyntheticDataset(test_paths, test_labels, transform=args.data_transforms['val'])

        # test dataloader with SinGAN
        if args.model == "SinGAN":
            test_ds = SinGANSyntheticDataset(test_paths, test_labels, transform=args.data_transforms['val'])

    # in this case the train dataloader is created from original dataset
    else:

        test_paths, test_labels = get_original_images(args)
        
        test_ds = OriginalDefectDataset(test_paths, test_labels, transform=args.data_transforms['val'])
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader

# training function 
def train(train_loader, val_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        # print epoch with the corrisponding metrics 
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
    
    # save the model in the args
    args.model_ = model

    print('Training completed')
    
# test function
def test(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    parser.add_argument("--cuda", action='store_true', help='Use CUDA for computation', default=False)
    parser.add_argument("--base_dir", type=str, help="Base directory of the projects", default="/content/mla_project/")
    parser.add_argument("--mode", type=str, help="GEN_train or GEN_test", required=True)
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--experiment", type=str, help="Experiment", required=True)
    parser.add_argument("--backbone", type=str, help="Backbone model", default="resnet50")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model", default=False)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=16)
    parser.add_argument("--val_split", type=float, help="Validation split", default=0.2)
    parser.add_argument("--num_workers", type=int, help="Number of workers", default=4)
    parser.add_argument("--random_seed", type=int, help="Random seed", default=42)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=20)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
    parser.add_argument('--checkpoint', type=str, default='best_model')

    args = parser.parse_args()

    # control if the mode is correct
    if args.mode not in ("GEN_train", "GEN_test"):
        print("Invalide mode. Choose either 'GEN_train' or 'GEN_test'")
        sys.exit()

    # control if the model is correct
    if args.model not in ("VAE", "CVAE", "GANs", "Diffusion", "SinGAN"):
        print("Invalid model. Choose either 'VAE', 'CVAE', 'Diffusion', 'SinGAN' or 'GANs'")
        sys.exit()

    # path of the generated data based on specific model and number of experiment
    args.dir_generated = os.path.join(args.base_dir, "images", "augmented", args.model, args.experiment)

    if not os.path.exists(args.dir_generated):
        print(f"Directory {args.dir_generated} doesn't exist. Please try again")
        sys.exit()

    # get transformations for the images
    args.data_transforms = get_transformations(args)

    if args.mode == "GEN_train":
        print("Train and Val dataloaders with original data...")
        train_loader, val_loader = get_train_val_dataloaders(args)
        print("Train and val dataloaders loaded!\n")

        print("Training with original data...")
        train(train_loader, val_loader, args)

        print(f"\nTest dataloader with {args.model} data...")
        test_loader = get_test_dataloader(args) # Get test data loader from CvaeSyntheticDataset
        print("Test dataloader loaded!\n")

        print(f"Test on generated data by {args.model}...")
        test(args.model_, test_loader)


    if args.mode == "GEN_test": 

        print(f"Train and Val dataloaders with {args.model} data...")
        train_loader, val_loader = get_train_val_dataloaders(args)
        print("Train and val dataloaders loaded!\n")
        
        print(f"Training with {args.model} synthetic data...")
        train(train_loader, val_loader, args)

        print(f"\nTest dataloader with original data...")
        test_loader = get_test_dataloader(args)
        print("Test dataloader loaded!\n")

        print(f"Test on original data...")
        test(args.model_, test_loader)

