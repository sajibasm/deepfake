import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from captum.attr import LayerGradCam

# Handle command line image input
if len(sys.argv) < 2:
    print("Usage: python gradcam.py path_to_image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

# Load model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model/resnet_model.pth", map_location='cpu'))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Predict
output = model(input_tensor)
pred_label = output.argmax(dim=1).item()

# Apply Grad-CAM
layer_gc = LayerGradCam(model, model.layer4)
attr = layer_gc.attribute(input_tensor, target=pred_label)
attr = torch.nn.functional.interpolate(attr, (224, 224), mode='bilinear')

# Convert CAM to numpy
def convert_to_numpy(tensor):
    array = tensor.squeeze().detach().numpy()
    if array.ndim == 2:
        return array  # For 2D heatmaps (H x W)
    return np.transpose(array, (1, 2, 0))  # For RGB-style arrays

# Plot
cam = convert_to_numpy(attr)

plt.imshow(image)
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.title(f"Grad-CAM - {'FAKE' if pred_label else 'REAL'}")
plt.axis('off')
plt.show()