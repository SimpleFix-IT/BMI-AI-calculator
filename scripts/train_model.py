import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader

# # ✅ Model Definition
# class HeightWeightEstimator(nn.Module):
#     def __init__(self):
#         super(HeightWeightEstimator, self).__init__()
#         self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # ✅ Updated weights argument

#         # ❌ Default ResNet18 last FC layer ko hatao
#         self.base_model.fc = nn.Identity()  # Remove fully connected layer

#         # ✅ Corrected FC layer (ResNet18 last layer 512 features return karta hai)
#         self.fc = nn.Linear(512, 2)  # 2 Outputs: Height & Weight

#     def forward(self, x):
#         x = self.base_model(x)  # ✅ ResNet ka last FC layer hata diya hai
#         print(f"ResNet Output Shape: {x.shape}")  # Debugging ke liye Print
#         x = self.fc(x)  # ✅ Now shape will be (batch_size, 2)
#         return x
class HeightWeightEstimator(nn.Module):
    def __init__(self):
        super(HeightWeightEstimator, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(512, 2)  # Attach FC layer to base_model

    def forward(self, x):
        x = self.base_model(x)  # Directly pass through base_model
        return x

# ✅ Dummy Dataset (Replace with Real Data)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = FakeData(size=100, transform=transform)  # 100 Fake Images
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)

# ✅ Model, Loss, and Optimizer
model = HeightWeightEstimator()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training Loop
print("🚀 Training Started...")
for epoch in range(5):  # Train for 5 epochs
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)

        print(f"Outputs shape: {outputs.shape}")  # Expected (batch_size, 2)
        print(f"Targets shape before reshape: {targets.shape}") 

        # ✅ Fix target shape
        targets = torch.stack((targets, targets), dim=1)  # Convert (batch_size,) → (batch_size, 2)

        print(f"Targets shape after reshape: {targets.shape}") 

        loss = criterion(outputs, targets.float())  # Ensure shape matches (batch_size, 2)

        loss.backward()
        optimizer.step()
    
    # ✅ Print loss per epoch
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# ✅ Save Model
model_path = "model_weights.pth"
torch.save(model.state_dict(), model_path)
print("✅ Training Complete! Model weights saved to 'model_weights.pth'")
