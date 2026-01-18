from typing import Tuple
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)

def build_model(name: str, num_classes: int, pretrained: bool) -> Tuple[nn.Module, str]:
    """
    Returns (model, tag). Tag encodes backbone + tl/scratch.
    """
    name = name.lower()
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        tag = f"resnet18_{'tl' if pretrained else 'scratch'}"
        return model, tag

    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        tag = f"effnet_b0_{'tl' if pretrained else 'scratch'}"
        return model, tag

    raise ValueError(f"Unknown model name: {name}")
