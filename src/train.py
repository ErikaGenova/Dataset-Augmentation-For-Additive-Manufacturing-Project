# training script (CLI interface)

import argparse
import torch
from torch import optim, nn
from src.data_loader import get_dataloaders
from src.model import build_model
import pandas as pd

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
    print(f"\nModel summary:\n{model}")

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train PBF defect detector")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--checkpoint', type=str, default='best_model.pth', help='Path to save best model')
    args = parser.parse_args()
    train(args)