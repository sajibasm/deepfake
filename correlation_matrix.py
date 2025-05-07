import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Paths
TEST_DIR = "data/test"
MODEL_DIR = "model"
MODEL_PATHS = {
    "efficientnet": os.path.join(MODEL_DIR, "best_efficientnetv2.pth"),
    "resnet": os.path.join(MODEL_DIR, "best_resnet50.pth"),
    "xception": os.path.join(MODEL_DIR, "best_xception_model.pth")
}

# Set device (MPS for Mac M3, CUDA for Nvidia, CPU otherwise)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data loader
val_dataset = ImageFolder(TEST_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load Models
models_dict = {
    "EfficientNetV2": models.efficientnet_v2_s(weights=None),
    "ResNet50": models.resnet50(weights=None),
    "Xception": timm.create_model("legacy_xception", pretrained=False, num_classes=1)
}

models_dict["EfficientNetV2"].classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(models_dict["EfficientNetV2"].classifier[-1].in_features, 1),
    nn.Sigmoid()
)

models_dict["ResNet50"].fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(models_dict["ResNet50"].fc.in_features, 1),
    nn.Sigmoid()
)

models_dict["Xception"].fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(models_dict["Xception"].get_classifier().in_features, 1),
    nn.Sigmoid()
)

# Load the trained weights
for model_name, model in models_dict.items():
    model.load_state_dict(torch.load(MODEL_PATHS[model_name.lower()]))
    model.to(DEVICE)
    model.eval()

# Extract model outputs
all_outputs = {name: [] for name in models_dict.keys()}

with torch.no_grad():
    for images, _ in tqdm(val_loader, desc="Extracting Model Outputs"):
        images = images.to(DEVICE)

        for model_name, model in models_dict.items():
            outputs = model(images).view(-1).cpu().numpy()
            all_outputs[model_name].extend(outputs)

# Convert to DataFrame for correlation analysis
df_outputs = pd.DataFrame(all_outputs)

# Generate and visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df_outputs.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of Model Outputs")
plt.show()
