import torch
import torch.nn as nn
import torchvision.models as models

class HeightWeightEstimator(nn.Module):
    def __init__(self):
        super(HeightWeightEstimator, self).__init__()
        
        # ✅ ResNet18 model load karna
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # ✅ Default 1000 output classes hoti hain, ise 2 outputs me convert karna
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, 2)  # Fix: Only 2 outputs (height, weight)

    def forward(self, x):
        return self.base_model(x)
