import torch
import torch.nn as nn
from torchvision import models

class MultiTaskResNet(nn.Module):
    def __init__(self, num_fruits=3, num_freshness=2):
        super(MultiTaskResNet, self).__init__()

        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        self.fruit_head = nn.Linear(in_features, num_fruits)
        self.freshness_head = nn.Linear(in_features, num_freshness)

    def forward(self, x):
        features = self.base_model(x)

        fruit_output = self.fruit_head(features)
        freshness_output = self.freshness_head(features)

        return fruit_output, freshness_output
