import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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

# Inference
all_labels, all_probs = [], []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating Weighted Ensemble"):
        images, labels = images.to(DEVICE), labels.to(DEVICE).float()

        # Weighted Ensemble Prediction
        efficientnet_outputs = efficientnet(images).view(-1) * weights["efficientnet"]
        resnet_outputs = resnet(images).view(-1) * weights["resnet"]
        xception_outputs = xception(images).view(-1) * weights["xception"]

        # Weighted average
        ensemble_outputs = (
            efficientnet_outputs +
            resnet_outputs +
            xception_outputs
        ) / sum(weights.values())

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(ensemble_outputs.cpu().numpy())

# Generate ROC and PR curves
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Weighted Ensemble')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "weighted_ensemble_roc_curve.png"))
plt.show()

precision, recall, _ = precision_recall_curve(all_labels, all_probs)
average_precision = average_precision_score(all_labels, all_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkgreen', lw=2, label=f"PR Curve (AP = {average_precision:.4f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Weighted Ensemble')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "weighted_ensemble_pr_curve.png"))
plt.show()

# Classification report
preds = (np.array(all_probs) > 0.5).astype(int)
print("\n📄 Classification Report:\n")
class_labels = val_dataset.classes
print(classification_report(all_labels, preds, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(all_labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Weighted Ensemble')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "weighted_ensemble_confusion_matrix.png"))
plt.show()
