import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_auc_score
from deap import base, creator, tools, algorithms
import random
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

# Load Models
models_dict = {
    "efficientnet": models.efficientnet_v2_s(weights=None),
    "resnet": models.resnet50(weights=None),
    "xception": timm.create_model("legacy_xception", pretrained=False, num_classes=1)
}

models_dict["efficientnet"].classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(models_dict["efficientnet"].classifier[-1].in_features, 1),
    nn.Sigmoid()
)

models_dict["resnet"].fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(models_dict["resnet"].fc.in_features, 1),
    nn.Sigmoid()
)

models_dict["xception"].fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(models_dict["xception"].get_classifier().in_features, 1),
    nn.Sigmoid()
)

# Load the trained weights
for model_name, model in models_dict.items():
    model.load_state_dict(torch.load(MODEL_PATHS[model_name]))
    model.to(DEVICE)
    model.eval()

# Prepare data for genetic algorithm
all_labels, all_model_outputs = [], {name: [] for name in models_dict.keys()}

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Extracting Model Outputs"):
        images = images.to(DEVICE)
        labels = labels.cpu().numpy()
        all_labels.extend(labels)

        for model_name, model in models_dict.items():
            outputs = model(images).view(-1).cpu().numpy()
            all_model_outputs[model_name].extend(outputs)

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_model_outputs = {name: np.array(outputs) for name, outputs in all_model_outputs.items()}

# Genetic Algorithm Setup
POP_SIZE = 20
GENS = 30
MUT_PROB = 0.2
CX_PROB = 0.5

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_weight", random.uniform, 0.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_weight, n=len(models_dict))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate_weights(weights):
    weights = np.array(weights)
    weights /= np.sum(weights)

    # Calculate weighted ensemble output
    ensemble_output = np.zeros_like(all_labels, dtype=float)
    for i, (name, outputs) in enumerate(all_model_outputs.items()):
        ensemble_output += weights[i] * outputs

    # Calculate AUC score
    auc_score = roc_auc_score(all_labels, ensemble_output)
    return auc_score,


toolbox.register("evaluate", evaluate_weights)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the genetic algorithm
population = toolbox.population(n=POP_SIZE)
algorithms.eaSimple(population, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB, ngen=GENS, verbose=True)

# Extract the best weights
best_individual = tools.selBest(population, k=1)[0]
best_weights = np.array(best_individual)
best_weights /= np.sum(best_weights)

# Map weights to model names
optimal_weights = {name: weight for name, weight in zip(models_dict.keys(), best_weights)}

print("\n✅ Optimal Weights Found:")
print(optimal_weights)
