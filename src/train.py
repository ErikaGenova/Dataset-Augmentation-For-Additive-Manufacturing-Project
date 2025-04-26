# training script (CLI interface)
import sys
import os

# Add path 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from torch import optim, nn
from data_loader import DefectDataset, data_transforms, get_all_data
from model import build_model
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from PIL import Image
import glob
import numpy as np


def train_one_fold(train_idx, val_idx, file_paths, labels, device, args, fold):
    '''Allena un fold del k-fold.'''
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
    
    # Freeze all layers except the last 2 layers
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        weight_decay=1e-3 # L2 regularization
    )

    best_val_acc = 0.0
    logs = []

    # Add scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    for epoch in range(args.epochs):
        # Training loop
        model.train()
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
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation loop
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
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        # Update learning rate
        scheduler.step(val_loss)

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

    for fold, (train_idx, val_idx) in enumerate(kf.split(file_paths, labels), 1):
        print(f"\n===== Fold {fold} =====")
        logs = train_one_fold(train_idx, val_idx, file_paths, labels, device, args, fold)
        all_logs += [[fold] + row for row in logs]

    # Save all logs to CSV
    df = pd.DataFrame(all_logs, columns=['fold', 'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    df.to_csv('kfold_logs.csv', index=False)
    print("\nK-Fold training completed. Metrics saved to 'kfold_logs.csv'.")


if __name__ == '__main__':
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