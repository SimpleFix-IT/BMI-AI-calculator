import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights  # Import weights properly

class HeightWeightEstimator(nn.Module):
    def __init__(self):
        super(HeightWeightEstimator, self).__init__()
        # Fix the deprecated warning by using weights properly
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(512, 2)  # Ensure same as training

    def forward(self, x):
        x = self.base_model(x)  # Directly pass through base_model
        return x

# Load the trained model
model = HeightWeightEstimator()
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
model.eval()

print("✅ Model loaded successfully!")

# Dummy input for checking
dummy_input = torch.randn(1, 3, 224, 224)  # Ensure input matches training size
output = model(dummy_input)

# Print output
print("Output shape:", output.shape)
print("Output values:", output.detach().numpy())  # Convert to NumPy for readability
