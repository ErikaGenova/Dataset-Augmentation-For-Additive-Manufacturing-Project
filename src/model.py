# builds & returns the CNN model

import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=2, backbone='resnet50', pretrained=True):
    '''Return a classification model with final layer adapted to num_classes.'''
    if backbone == 'resnet50':
        model = models.resnet50(weights=pretrained)
        # Adapt first conv if grayscale input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        # Add dropout before the final layer
        model.fc = nn.Sequential(
            nn.Dropout(0.5),         # Aggiunto Dropout
            nn.Linear(in_features, num_classes)
        )
    elif backbone == 'resnet34':
        model = models.resnet34(weights=pretrained)
        # Adapt first conv if grayscale input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        # Add dropout before the final layer
        model.fc = nn.Sequential(
            nn.Dropout(0.5),         # Aggiunto anche qui
            nn.Linear(in_features, num_classes)
        )
    elif backbone == 'resnet18':
        model = models.resnet18(weights=pretrained)
        # Adapt first conv if grayscale input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        # Add dropout before the final layer
        model.fc = nn.Sequential(
            nn.Dropout(0.5),         # Aggiunto anche qui
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model