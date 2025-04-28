### FILE: predict.py
import sys
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models import ResNet18_Weights

if len(sys.argv) < 2:
    print("Usage: python predict.py path_to_image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model/best_resnet_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    label = "FAKE" if predicted.item() == 1 else "REAL"
    print(f"Prediction: {label}")


