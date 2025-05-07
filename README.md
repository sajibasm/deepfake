# 🧠 DeepGuard Image Detection

This project uses Convolutional Neural Networks (CNNs) to detect deepfake images (fake vs. real) based on facial image classification using PyTorch. The models include **ResNet50**, **EfficientNetB5**, and **Xception**, each optimized for MPS (Apple Silicon), CUDA (NVIDIA), and CPU execution. An ensemble model is also included, which finds the **optimal weight** for combining the predictions of these models for more accurate and robust classification.

---

## 📁 Dataset Structure

Ensure your data is organized as follows:

```
data/
├── train/
│   ├── fake/
│   └── real/
└── test/
    ├── fake/
    └── real/
```

- **train/fake/** — Contains training images of fake faces.
- **train/real/** — Contains training images of real faces.
- **test/fake/** — Contains test images of fake faces.
- **test/real/** — Contains test images of real faces.

---

## 📦 Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Models

Each model script is optimized for both GPU (CUDA/MPS) and CPU training:

- **EfficientNetB5**: Run `efficientnet.py`
- **ResNet50**: Run `resnet50.py`
- **Xception**: Run `xception.py`

For example, to train the ResNet50 model:

```bash
python3 resnet50.py
```

---

## 🤖 Ensemble Models for Final Prediction

### **Optimal Weight Ensemble**

Uses an optimal weight search algorithm to find the best combination of model weights for the final prediction:

```bash
python3 ensemble_optimal_weight.py
```

### **Basic Ensemble Prediction**

Combines predictions from ResNet50, EfficientNetB0, and Xception for more robust classification:

```bash
python3 ensemble.py
```

Ensure all three models are trained and their weights are saved in the **model/** directory before running these ensemble scripts.

### ** Single Image Ensemble Prediction**

To make a single image prediction with the ensemble, you can use the `predict.py` script:

```bash
python3 predict.py
```
---

## 📊 Model Evaluation

Confusion matrices and classification reports are automatically generated after training. Check the **model/** directory for saved models and evaluation plots.

---

## 📈 Results and Logs

All training logs, loss curves, and confusion matrices will be saved in the **model/** directory. You can visualize them to track model performance over time.

---

## 🛠️ Future Improvements
- Face Cropping and Preprocessing.
- Vision Transformer (ViT) Integration.
- Integrate tools like Grad-CAM or LIME to visualize which parts of the image influence model decisions.
- Automated Hyperparameter Tuning.
- Add mixed precision training for faster performance.
- Implement early stopping and learning rate scheduling.
- Optimize data loading for larger datasets.

---

## 🤖 Contributing

Feel free to submit a pull request if you have improvements or bug fixes.

---

## 📄 License
This project is licensed under the MIT License.
