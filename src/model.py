# builds & returns the CNN model
import torch
import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=2, backbone='resnet50', pretrained=False):
    '''Return a classification model with final layer adapted to num_classes.'''
    if backbone == 'resnet50':
        model = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
        #print("Weights model: ", model.conv1.weight)

        # Adapt first conv if grayscale input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == 'resnet34':
        model = models.resnet34(weights=None if not pretrained else models.ResNet34_Weights.DEFAULT)
        #print("Weights model: ", model.conv1.weight)

        # Adapt first conv if grayscale input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == 'resnet18':
        model = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        #print("Weights model: ", model.conv1.weight)

        # Adapt first conv if grayscale input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    

    model_pretrained = build_model(num_classes=2, backbone='resnet18', pretrained=True)
    # Confronta i pesi del primo livello convoluzionale
    are_weights_equal = torch.equal(model.conv1.weight, model_pretrained.conv1.weight)

    print("\nI pesi del primo livello convoluzionale sono uguali?:", are_weights_equal)
    return model