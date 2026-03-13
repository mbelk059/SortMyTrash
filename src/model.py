import torch
import torch.nn as nn
from torchvision import models


def get_backbone(backbone_name="resnet18", pretrained=True):
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        return model, in_features
    if backbone_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        return model, in_features
    if backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, in_features
    raise ValueError(f"Unsupported backbone: {backbone_name}")


class WasteClassifier(nn.Module):
    def __init__(self, num_classes, backbone_name="resnet18", dropout=0.5, pretrained=True):
        super().__init__()
        backbone, in_features = get_backbone(backbone_name, pretrained=pretrained)
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out


def load_checkpoint(model, checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
