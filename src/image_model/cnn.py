import torch
import torch.nn as nn
import torchvision.models as models


class WatchCNN(nn.Module):
    """CNN model for luxury watch image classification"""

    def __init__(self, num_classes=10, pretrained=True):
        super(WatchCNN, self).__init__()

        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)

        # Replace final classification layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass through CNN"""
        return self.backbone(x)

    def extract_features(self, x):
        """Extract features before final classification"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x