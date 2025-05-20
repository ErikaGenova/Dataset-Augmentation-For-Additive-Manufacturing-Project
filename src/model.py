# builds & returns the CNN model
import torch
import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=2, backbone='resnet50', pretrained=False):
    '''Return a classification model with final layer adapted to num_classes.'''
    if backbone == 'resnet50':
        if not pretrained:
            print("No pretrained weights")
            model = models.resnet50(weights=None)
        else:
            print("Pretrained weights")
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Adapt first conv if grayscale input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == 'resnet34':
        if not pretrained:
            print("No pretrained weights")
            model = models.resnet34(weights=None)
        else:
            print("Pretrained weights")
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # Adapt first conv if grayscale input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == 'resnet18':
        if not pretrained:
            print("No pretrained weights")
            model = models.resnet18(weights=None)
        else:
            print("Pretrained weights")
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adapt first conv if grayscale input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model