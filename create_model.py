import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4  # Bolt, locatingpin, nuts, washers

# Load base model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# (Optional) Simulate training by randomizing weights
for param in model.parameters():
    if param.requires_grad:
        nn.init.normal_(param, mean=0.0, std=0.01)

# Save the dummy model
torch.save(model, "model.pth")
print("✅ Dummy model.pth created!")
