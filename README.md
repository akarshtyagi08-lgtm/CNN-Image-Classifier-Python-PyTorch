# ✍️ MNIST Handwritten Digit Recognition (PyTorch CNN)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Torchvision-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white" alt="Torchvision" />
  <img src="https://img.shields.io/badge/Accuracy-98.69%25-success?style=for-the-badge" alt="Accuracy" />
  <img src="https://img.shields.io/badge/Dataset-MNIST-blue?style=for-the-badge" alt="MNIST" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge" alt="Status" />
</p>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" width="500" alt="MNIST Handwritten Digits Sample" />
</p>

A deep learning project implementing a Convolutional Neural Network (CNN) built with **PyTorch** to classify handwritten digits (0 through 9) using the **MNIST** dataset[span_0](start_span)[span_0](end_span). The model achieves a **98.69% test accuracy** after 50 epochs of training[span_1](start_span)[span_1](end_span).

---

## 📌 Project Overview & Specifications

* 🖼️ **Dataset:** MNIST Grayscale Handwritten Digits (28x28 pixels)[span_2](start_span)[span_2](end_span)
* 🏷️ **Classes:** 10 classes (`0` to `9`)[span_3](start_span)[span_3](end_span)
* 🧠 **Network Type:** 2D Convolutional Neural Network (CNN)[span_4](start_span)[span_4](end_span)
* 📉 **Loss Function:** `nn.CrossEntropyLoss()`[span_5](start_span)[span_5](end_span)
* ⚙️ **Optimization Algorithm:** Adam Optimizer ($lr = 0.001$)[span_6](start_span)[span_6](end_span)
* 📦 **Batch Size:** 64[span_7](start_span)[span_7](end_span)
* 🔁 **Total Epochs:** 50[span_8](start_span)[span_8](end_span)
* 🎯 **Final Testing Accuracy:** **98.69%**
* 💾 **Model Weights:** Saved locally as `model.pth`[span_9](start_span)[span_9](end_span)

---

## 📦 Required Libraries & Dependencies

Ensure your environment satisfies the following package requirements:

* `python >= 3.8`
* `torch` (Core PyTorch tensor computation and neural network modules)
* `torchvision` (Datasets, transforms, and utilities for computer vision)

To install all required libraries, run:

```bash
pip install torch torchvision
```

---

## 🧠 Detailed Model Architecture

The `SimpleCNN` network architecture processes single-channel $28 \times 28$ image tensors through feature extraction and linear classification layers[span_10](start_span)[span_10](end_span):

```text
Input Tensor (Batch, 1, 28, 28)
│
├── [Layer 1] Conv2d (in=1, out=8, kernel=3, padding=1) ──> (Batch, 8, 28, 28)
├── [Activation] ReLU()
├── [Pooling] MaxPool2d (kernel=2) ─────────────────────────> (Batch, 8, 14, 14)
│
├── [Layer 2] Conv2d (in=8, out=16, kernel=3, padding=1) ─> (Batch, 16, 14, 14)
├── [Activation] ReLU()
├── [Pooling] MaxPool2d (kernel=2) ─────────────────────────> (Batch, 16, 7, 7)
│
├── [Flattening] view(batch_size, 16 * 7 * 7) ──────────────> (Batch, 784)
│
├── [Classifier 1] Linear (in=784, out=64) ─────────────────> (Batch, 64)
├── [Activation] ReLU()
└── [Classifier 2] Linear (in=64, out=10) ──────────────────> (Batch, 10) [Logits]
```

### Layer Summary Table

| Stage | Layer Type | Input Dimension | Output Dimension | Activation / Kernel |
| :--- | :--- | :--- | :--- | :--- |
| **Feature Extractor** | `Conv2d` | `1 x 28 x 28` | `8 x 28 x 28` | Kernel: `3x3`, Padding: `1`[span_11](start_span)[span_11](end_span) |
| **Downsampling 1** | `MaxPool2d` | `8 x 28 x 28` | `8 x 14 x 14` | Kernel: `2x2`[span_12](start_span)[span_12](end_span) |
| **Feature Extractor** | `Conv2d` | `8 x 14 x 14` | `16 x 14 x 14` | Kernel: `3x3`, Padding: `1`[span_13](start_span)[span_13](end_span) |
| **Downsampling 2** | `MaxPool2d` | `16 x 14 x 14` | `16 x 7 x 7` | Kernel: `2x2`[span_14](start_span)[span_14](end_span) |
| **Flatten** | `Tensor View` | `16 x 7 x 7` | `784` | Flat vector[span_15](start_span)[span_15](end_span) |
| **Dense Layer 1** | `Linear` | `784` | `64` | `ReLU`[span_16](start_span)[span_16](end_span) |
| **Output Layer** | `Linear` | `64` | `10` | 10-Class Logits[span_17](start_span)[span_17](end_span) |

---

## 📊 Training Logs & Performance Report

Complete terminal training logs across all 50 epochs:

```text
Epoch: 1/50  | Loss: 322.0067
Epoch: 2/50  | Loss: 90.0987
Epoch: 3/50  | Loss: 64.0636
Epoch: 4/50  | Loss: 51.8399
Epoch: 5/50  | Loss: 41.4909
Epoch: 6/50  | Loss: 36.0349
Epoch: 7/50  | Loss: 30.5107
Epoch: 8/50  | Loss: 27.0305
Epoch: 9/50  | Loss: 24.4495
Epoch: 10/50 | Loss: 21.0808
Epoch: 11/50 | Loss: 19.2566
Epoch: 12/50 | Loss: 16.3445
Epoch: 13/50 | Loss: 15.3522
Epoch: 14/50 | Loss: 13.0162
Epoch: 15/50 | Loss: 11.6650
Epoch: 16/50 | Loss: 11.1970
Epoch: 17/50 | Loss: 10.2069
Epoch: 18/50 | Loss: 9.2191
Epoch: 19/50 | Loss: 8.7758
Epoch: 20/50 | Loss: 7.7354
Epoch: 21/50 | Loss: 7.9956
Epoch: 22/50 | Loss: 7.8711
Epoch: 23/50 | Loss: 6.6881
Epoch: 24/50 | Loss: 5.1434
Epoch: 25/50 | Loss: 5.9346
Epoch: 26/50 | Loss: 5.2848
Epoch: 27/50 | Loss: 4.6919
Epoch: 28/50 | Loss: 4.4404
Epoch: 29/50 | Loss: 4.2653
Epoch: 30/50 | Loss: 5.3771
Epoch: 31/50 | Loss: 4.0351
Epoch: 32/50 | Loss: 4.4858
Epoch: 33/50 | Loss: 3.5276
Epoch: 34/50 | Loss: 4.4886
Epoch: 35/50 | Loss: 3.5344
Epoch: 36/50 | Loss: 4.0571
Epoch: 37/50 | Loss: 3.4165
Epoch: 38/50 | Loss: 3.9304
Epoch: 39/50 | Loss: 4.7492
Epoch: 40/50 | Loss: 2.1381
Epoch: 41/50 | Loss: 3.5422
Epoch: 42/50 | Loss: 3.6457
Epoch: 43/50 | Loss: 1.0554
Epoch: 44/50 | Loss: 2.6497
Epoch: 45/50 | Loss: 5.2746
Epoch: 46/50 | Loss: 1.6799
Epoch: 47/50 | Loss: 4.5540
Epoch: 48/50 | Loss: 1.3626
Epoch: 49/50 | Loss: 3.7569
Epoch: 50/50 | Loss: 2.7488
Accuracy in Testing: 98.69%
Model Saved as model.pth
```

---

## 🚀 How to Run & Use

### 1. Training and Evaluation Pipeline
Execute the training script to fetch the MNIST dataset, train across 50 epochs, compute evaluation metrics, and export the trained model state:

```bash
python train.py
```

### 2. Inference / Model Loading Script
Load the saved state dictionary (`model.pth`) and perform sample classification:

```python
import torch
from train import SimpleCNN

# Setup compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize architecture and load weights
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

print("PyTorch model loaded successfully and ready for evaluation!")
```
