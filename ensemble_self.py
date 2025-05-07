import os
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Paths
MODEL_DIR = "model"
IMAGE_PATH = "test_images/your_fake_image.jpg"  # Replace with your image path
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

# Confidence Scoring Function
def calculate_confidence(output):
    confidence = abs((output - 0.5) * 200)  # Scale to 0-100%
    return confidence

# Model Weights (Adjust based on validation accuracy or other metrics)
weights = {
    "efficientnet": 0.4,
    "resnet": 0.3,
    "xception": 0.3
}

# Model 1 - EfficientNetV2
efficientnet = models.efficientnet_v2_s(weights=None)
efficientnet.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(efficientnet.classifier[-1].in_features, 1),
    nn.Sigmoid()
)
efficientnet.load_state_dict(torch.load(MODEL_PATHS["efficientnet"]))
efficientnet.to(DEVICE)
efficientnet.eval()

# Model 2 - ResNet50
resnet = models.resnet50(weights=None)
resnet.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(resnet.fc.in_features, 1),
    nn.Sigmoid()
)
resnet.load_state_dict(torch.load(MODEL_PATHS["resnet"]))
resnet.to(DEVICE)
resnet.eval()

# Model 3 - Xception
xception = timm.create_model("legacy_xception", pretrained=False, num_classes=1)
xception.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(xception.get_classifier().in_features, 1),
    nn.Sigmoid()
)
xception.load_state_dict(torch.load(MODEL_PATHS["xception"]))
xception.to(DEVICE)
xception.eval()

# Load and preprocess the image
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# Ensemble prediction with weighted voting
with torch.no_grad():
    efficientnet_output = efficientnet(input_tensor).view(-1).item()
    resnet_output = resnet(input_tensor).view(-1).item()
    xception_output = xception(input_tensor).view(-1).item()

    # Weighted average
    ensemble_output = (
        weights["efficientnet"] * efficientnet_output +
        weights["resnet"] * resnet_output +
        weights["xception"] * xception_output
    ) / sum(weights.values())

# Print individual model and ensemble outputs
print(f"🔍 EfficientNetV2 Output: {efficientnet_output:.4f} (Weight: {weights['efficientnet']})")
print(f"🔍 ResNet50 Output: {resnet_output:.4f} (Weight: {weights['resnet']})")
print(f"🔍 Xception Output: {xception_output:.4f} (Weight: {weights['xception']})")
print(f"📝 Weighted Ensemble Output: {ensemble_output:.4f}")

# Thresholding for classification
threshold = 0.5
prediction = "FAKE" if ensemble_output > threshold else "REAL"

# Confidence scoring
confidence = calculate_confidence(ensemble_output)

# Display result with confidence
print(f"\n🎯 Predicted Class: {prediction} ({confidence:.2f}% confident)")

# Visualize the result
plt.imshow(image)
plt.title(f"Predicted: {prediction} ({confidence:.2f}% confident)")
plt.axis("off")
plt.show()
