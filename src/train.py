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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from PIL import Image
import glob
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


def get_class_weights(labels):
    '''Compute class weights for imbalanced dataset.'''
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    class_weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
    class_weights = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.torch.float32)
    return class_weights

def plot_confusion_matrix(y_true, y_pred, fold, epoch):
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(f"Confusion Matrix - Fold {fold} (Validation)")

    # Save the confusion matrix as an image
    plt.savefig(f"confusion_matrix_fold{fold}.png")
    plt.close(fig)  # Close the figure to avoid blocking

    print(f"Confusion matrix for Fold {fold} (Validation) saved as 'confusion_matrix_fold{fold}.png'")
    return cm
    """

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n     Confusion Matrix - Fold {fold}  - Epoch {epoch+1} (Validation):")
     # print the confusion matrix with 5 spaces indentation
    indent = " " * 5  
    for row in cm:
        print(f"{indent}{row}")
    print("\n")
    return cm

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n Loading data...")
    # Get train and validation loaders with val_split
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        test=args.test,
        test_split=args.test_split
    )
    print("\nData loaded!")
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

    if args.test:
        # Test set evaluation
        print("\nEvaluating on the test set...")
        test_running_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        test_loss = test_running_loss / test_total
        test_acc = test_correct / test_total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


def train_one_fold(train_idx, val_idx, file_paths, labels, device, args, fold, class_weights, aug=False, test_paths=None, test_labels=None):
    '''Train one fold of the K-Fold cross-validation.'''
    # Dataset and loader
    train_dataset = DefectDataset([file_paths[i] for i in train_idx],
                                   [labels[i] for i in train_idx],
                                   transform=data_transforms['train_aug'] if aug else data_transforms['train'])

    val_dataset = DefectDataset([file_paths[i] for i in val_idx],
                                 [labels[i] for i in val_idx],
                                 transform=data_transforms['val'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"\n[Fold {fold}] Number of training images: {len(train_dataset)}")
    print(f"[Fold {fold}] Number of validation images: {len(val_dataset)}")
    # Model
    model = build_model(backbone=args.backbone, pretrained=args.pretrained)
    model.to(device)

    # Freeze all layers except the last fully connected layer and layer4 to avoid overfitting
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
        #print("\n-- train --")
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
        #print("Predictions: ", pred_tot)
        #print("Labels: ", label_tot)

        # Validation loop
        #print("\n-- val --")
        pred_tot = []
        label_tot = []
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        all_targets, all_preds = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                # for debugging
                pred_tot.extend([p.item() for p in preds])
                label_tot.extend([t.item() for t in targets])
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        #print("Predictions: ", pred_tot)
        #print("Labels: ", label_tot)

        # Calculate precision, recall, and F1-score
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

        # Calculate per class metrics
        precision_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)
        
        # Update learning rate
        scheduler.step()

        # Log metrics (save global metrics but print per class metrics)
        logs.append([epoch+1, train_loss, val_loss, train_acc, val_acc, precision, recall, f1])
        print(f"[Fold {fold}] Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        """
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            if i == 0:
                print(f"     Class {i} (NoDefects): Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f} (No Defects)")
            else:
                print(f"     Class {i} (Defects): Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f} (Defects)")
        """
        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the checkpoint in the specified output directory
            checkpoint_path = os.path.join(args.output_dir, f"{args.checkpoint}_fold{fold}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            
        # Plot Validation Confusion matrix
        #plot_confusion_matrix(all_targets, all_preds, fold, epoch)

    # Test set evaluation (if test set is provided)
    if test_paths and test_labels:
        print(f"\n[Fold {fold}] Evaluating on the test set...")
        test_dataset = DefectDataset(test_paths, test_labels, transform=data_transforms['val'])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f"[Fold {fold}] Number of test images: {len(test_dataset)}")

        test_running_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        test_loss = test_running_loss / test_total
        test_acc = test_correct / test_total
        print(f"[Fold {fold}] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    return logs


def train_kfold(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    file_paths, labels = get_all_data(args.data_dir)
    labels = np.array(labels)

    # Reserve a test set if args.test is True
    if args.test:
        train_val_idx, test_idx = train_test_split(
            list(range(len(file_paths))),
            test_size=args.test_split,
            stratify=labels,
            random_state=42
        )
        test_paths = [file_paths[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        file_paths = [file_paths[i] for i in train_val_idx]
        labels = [labels[i] for i in train_val_idx]
    else:
        test_paths, test_labels = None, None

    kf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    all_logs, fold_metrics = [], []

    # Compute class weights
    class_weights = get_class_weights(labels)

    # check augmentation
    aug = True if args.aug == 'True' else False

    for fold, (train_idx, val_idx) in enumerate(kf.split(file_paths, labels), 1):
        print(f"\n===== Fold {fold} =====")
        logs = train_one_fold(train_idx, val_idx, file_paths, labels, device, args, fold, class_weights, aug, test_paths, test_labels)
        all_logs += [[fold] + row for row in logs]
        
        # Save metrics of the last epoch for this fold
        last_epoch_metrics = logs[-1]  # Metrics of the last epoch
        fold_metrics.append(last_epoch_metrics[2:7])  # Extract exactly [val_loss, val_acc, precision, recall, f1]

    # Convert fold_metrics to a NumPy array for averaging
    fold_metrics = np.array(fold_metrics, dtype=float)

    # Calculate average metrics across folds
    avg_metrics = np.mean(fold_metrics, axis=0)
    avg_val_loss, avg_val_acc, avg_precision, avg_recall, avg_f1 = avg_metrics

    print(f"\n===== Overall Metrics Across {args.k_folds} Folds =====")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Average Validation Accuracy: {avg_val_acc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")

    # Save all logs to CSV
    df = pd.DataFrame(all_logs, columns=['fold', 'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'precision', 'recall', 'f1'])
    df.to_csv('kfold_logs.csv', index=False)
    print("\nK-Fold training completed. Metrics saved to 'kfold_logs.csv'.")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train PBF defect detector")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--checkpoint', type=str, default='best_model')
    parser.add_argument('--aug', type=str, default='False', help='Use data augmentation')
    parser.add_argument('--pretrained', type=str, help='Use pretrained weights', default="True")
    parser.add_argument('--k-folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--is_kfold', type=str, default='True', help='Use K-Fold cross-validation')
    parser.add_argument('--test', type=str, default='True', help='Use test set for evaluation') # If this is true, a test set will be used 
    parser.add_argument('--test-split', type=float, default=0.2, help='Test split ratio') # If this is true, a test set will be used
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--output-dir', type=str, default='resnet_checkpoints', help='Directory to save checkpoints')  # New argument
    args = parser.parse_args()

    """
    To perform K-Fold cross-validation, set --is_kfold to True and specify the number of folds with --k-folds.
    To perform a single train/val split, set --is_kfold to False and specify the validation split ratio with --val-split.

    In both cases, to perform also testing, set --test to True and specify the test split ratio with --test-split.
    """
    # Convert is_kfold argument to boolean
    args.is_kfold = True if args.is_kfold.lower() == 'true' else False
    args.test = True if args.test.lower() == 'true' else False

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.is_kfold:
        print("Training with K-Fold cross-validation")
        train_kfold(args)
    else:
        print("Training with single train/val split")
        train(args)