# training script (CLI interface)
import sys
import os

# Add path 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from torch import optim, nn
from data_loader import get_dataloaders, DefectDataset, data_transforms, get_all_data
from model import build_model
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from PIL import Image
import glob
import numpy as np
from collections import Counter

def get_class_weights(labels):
    '''Compute class weights for imbalanced dataset.'''
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    class_weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
    class_weights = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.torch.float32)
    return class_weights

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n Loading data...")
    # Get train and validation loaders with val_split
    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    print("\nData loaded!")
    print(f"\nTrain samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

    # Build model
    print(f"\nUsing {args.backbone} as backbone")
    model = build_model(backbone=args.backbone, pretrained=True)
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


def train_one_fold(train_idx, val_idx, file_paths, labels, device, args, fold, class_weights):
    '''Train one fold of the K-Fold cross-validation.'''
    # Dataset and loader
    train_dataset = DefectDataset([file_paths[i] for i in train_idx],
                                   [labels[i] for i in train_idx],
                                   transform=data_transforms['train'])

    val_dataset = DefectDataset([file_paths[i] for i in val_idx],
                                 [labels[i] for i in val_idx],
                                 transform=data_transforms['val'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = build_model(backbone=args.backbone, pretrained=True)
    model.to(device)

    # Freeze all layers except the last fully connected layer to avoid overfitting
    """
    for name, param in model.named_parameters():
        if not name.startswith('fc'):
            param.requires_grad = False
    """
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4) # L2 regularization

    best_val_acc = 0.0
    logs = []

    # Add scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(args.epochs):
        # Training loop
        print("\n-- train --")
        model.train()
        pred_tot = []
        label_tot = []
        running_loss, correct, total = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            pred_tot.extend([p.item() for p in preds])
            label_tot.extend([t.item() for t in targets])
        train_loss = running_loss / total
        train_acc = correct / total
        print("Predictions: ", pred_tot)
        print("Labels: ", label_tot)

        # Validation loop
        print("\n-- val --")
        pred_tot = []
        label_tot = []
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                pred_tot.extend([p.item() for p in preds])
                label_tot.extend([t.item() for t in targets])
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        print("Predictions: ", pred_tot)
        print("Labels: ", label_tot)

        # Update learning rate
        scheduler.step()

        # Log metrics
        logs.append([epoch+1, train_loss, val_loss, train_acc, val_acc])
        print(f"[Fold {fold}] Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{args.checkpoint}_fold{fold}.pth")

    return logs


def train_kfold(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    file_paths, labels = get_all_data(args.data_dir)
    labels = np.array(labels)
    kf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    all_logs = []

    # Compute class weights
    class_weights = get_class_weights(labels)

    for fold, (train_idx, val_idx) in enumerate(kf.split(file_paths, labels), 1):
        print(f"\n===== Fold {fold} =====")
        logs = train_one_fold(train_idx, val_idx, file_paths, labels, device, args, fold, class_weights)
        all_logs += [[fold] + row for row in logs]

    # Save all logs to CSV
    df = pd.DataFrame(all_logs, columns=['fold', 'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    df.to_csv('kfold_logs.csv', index=False)
    print("\nK-Fold training completed. Metrics saved to 'kfold_logs.csv'.")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser("Train PBF defect detector")
    # parser.add_argument('--data-dir', type=str, required=True)
    # parser.add_argument('--batch-size', type=int, default=16)
    # parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    # parser.add_argument('--epochs', type=int, default=20)
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--backbone', type=str, default='resnet50')
    # parser.add_argument('--num-workers', type=int, default=4)
    # parser.add_argument('--checkpoint', type=str, default='best_model.pth', help='Path to save best model')
    # args = parser.parse_args()
    # train(args)
    parser = argparse.ArgumentParser("Train PBF defect detector with K-Fold")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--checkpoint', type=str, default='best_model')
    parser.add_argument('--k-folds', type=int, default=5, help='Number of cross-validation folds')
    args = parser.parse_args()

    train_kfold(args)