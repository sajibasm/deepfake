import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Paths
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_resnet50.pth")
EARLY_STOP_PATH = os.path.join(MODEL_DIR, "best_resnet50.pth")
os.makedirs(MODEL_DIR, exist_ok=True)

# Parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10
PATIENCE = 5
LEARNING_RATE = 0.001
MIN_DELTA = 0.001

# Set device (MPS for Mac M3, CUDA for Nvidia, CPU otherwise)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Data transformations
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data loaders
train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = ImageFolder(TEST_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model setup (ResNet50)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid()
)
model.to(DEVICE)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Mixed Precision (AMP) Scaler
scaler = torch.cuda.amp.GradScaler()

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"✅ New best model saved with validation loss: {val_loss:.4f}")
        else:
            self.counter += 1
            print(f"⚠️ No improvement. Early stopping counter: {self.counter}/{self.patience}")

        return self.counter >= self.patience

# Initialize EarlyStopping
early_stopper = EarlyStopping(patience=PATIENCE, save_path=EARLY_STOP_PATH)

# Training loop with AMP and early stopping
print("🚀 Training ResNet50 with Mixed Precision (AMP) and Early Stopping...")
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct = 0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=DEVICE != torch.device("cpu")):
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)

        # Mixed precision training
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_correct += ((outputs > 0.5).int() == labels.int()).sum().item()
        progress_bar.set_postfix(loss=loss.item(), acc=total_correct / len(train_loader.dataset))

    train_loss = total_loss / len(train_dataset)
    train_acc = total_correct / len(train_dataset)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # Early stopping
    model.eval()
    val_loss, val_correct = 0, 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float()
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            val_loss += loss.item() * images.size(0)
            val_correct += ((outputs > 0.5).int() == labels.int()).sum().item()

    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)
    print(f"📝 Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    scheduler.step(val_loss)

    if early_stopper(val_loss, model):
        print("🛑 Early stopping triggered.")
        break

# Evaluation - ROC and PR curves
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - ResNet50')
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, "resnet50_roc_curve.png"))
plt.show()

precision, recall, _ = precision_recall_curve(all_labels, all_probs)
average_precision = average_precision_score(all_labels, all_probs)

plt.figure()
plt.plot(recall, precision, label=f'PR Curve (AP = {average_precision:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - ResNet50')
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, "resnet50_pr_curve.png"))
plt.show()

# Classification report
preds = (np.array(all_probs) > 0.5).astype(int)
print(classification_report(all_labels, preds))

# Confusion matrix
cm = confusion_matrix(all_labels, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - ResNet50')
plt.savefig(os.path.join(MODEL_DIR, "resnet50_confusion_matrix.png"))
plt.show()
