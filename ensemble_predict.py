import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import timm
import sys

# Check CLI input
if len(sys.argv) < 2:
    print("Usage: python ensemble_predict.py path_to_image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 2)
resnet.load_state_dict(torch.load("model/resnet_model.pth", map_location=device))
resnet = resnet.to(device)
resnet.eval()

# Load EfficientNet-B0
effnet = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
effnet.load_state_dict(torch.load("model/efficientnet_model.pth", map_location=device))
effnet = effnet.to(device)
effnet.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Predict using both models
with torch.no_grad():
    res_out = torch.nn.functional.softmax(resnet(input_tensor), dim=1)
    eff_out = torch.nn.functional.softmax(effnet(input_tensor), dim=1)

# Average predictions (ensemble)
combined = (res_out + eff_out) / 2
confidence, pred_class = torch.max(combined, 1)

# Map prediction to label
class_names = ['real', 'fake']
label = class_names[pred_class.item()]
print(f"Ensemble Prediction: {label.upper()} (Confidence: {confidence.item():.2f})")
