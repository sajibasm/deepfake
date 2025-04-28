# 🧠 Deepfake Image Detector

This project uses Convolutional Neural Networks (CNNs) to detect deepfake images (fake vs. real) based on facial image classification using PyTorch and the **Real and Fake Face Detection** dataset from Kaggle.

---

## 📁 Dataset

**Source**: [Real and Fake Face Detection - Kaggle](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)

- `data/real/` — Contains genuine facial images.
- `data/fake/` — Contains AI-generated or manipulated facial images.
- Format: JPG images.

---

## 🚀 Project Structure
data/ ├── real/ └── fake/

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python train.py

python predict.py path/to/image.jpg
