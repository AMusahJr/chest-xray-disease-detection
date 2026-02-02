

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_model(pretrained=True):
    """
    Returns a ResNet-50 model for binary classification.

    Args:
        pretrained (bool): If True, use ImageNet weights. If False, initialize randomly.

    Returns:
        torch.nn.Module: ResNet-50 with modified final layer.
    """
    if pretrained:
        try:
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
        except Exception as e:
            print("Could not load pretrained weights, initializing randomly:", e)
            model = resnet50(weights=None)
    else:
        model = resnet50(weights=None)

    # Replace the final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
