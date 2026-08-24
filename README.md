# ✍️ MNIST Handwritten Digit Recognition (PyTorch CNN)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Torchvision-Dataset-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white" alt="Torchvision" />
  <img src="https://img.shields.io/badge/Accuracy-98.69%25-success?style=for-the-badge" alt="Accuracy" />
  <img src="https://img.shields.io/badge/Architecture-Custom%20CNN-blueviolet?style=for-the-badge" alt="CNN" />
  <img src="https://img.shields.io/badge/Dataset-MNIST-blue?style=for-the-badge" alt="MNIST" />
</p>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" width="550" alt="MNIST Handwritten Digits Sample" />
</p>

A high-performance deep learning pipeline implementing an end-to-end 2D Convolutional Neural Network (CNN) built with **PyTorch** to classify handwritten digits (0 through 9) from the standard **MNIST** database. The model reaches **98.69% test accuracy** after 50 epochs of convergence.

---

## 📌 Project Overview & Technical Specifications

* 🖼️ **Dataset:** MNIST Grayscale Handwritten Digits (28x28 single-channel images)
* 🏷️ **Number of Classes:** 10 discrete classes (`0` to `9`)
* 🧠 **Model Family:** 2D Convolutional Neural Network (CNN)
* 📉 **Loss Criterion:** `nn.CrossEntropyLoss()`
* ⚙️ **Optimization Routine:** Adam Optimizer (`lr = 0.001`)
* 📦 **Batch Size:** 64 samples per batch
* 🔁 **Training Duration:** 50 Full Epochs
* 🎯 **Final Test Accuracy:** **98.69%**
* 💾 **Output Checkpoint:** Saved model weights to `model.pth`

---

## 📦 Required Libraries & Dependencies

Ensure your Python runtime environment has the required packages installed:

* `python >= 3.8`
* `torch` (Core PyTorch library for autograd and neural network layers)
* `torchvision` (Computer vision datasets and tensor transforms)

To install all dependencies with pip, run:

```bash
pip install torch torchvision
```

---

## 🧠 Detailed Model Architecture

The `SimpleCNN` model processes single-channel 28x28 pixel arrays by extracting hierarchical spatial representations through 2D convolutions, applying spatial downsampling via 2x2 max pooling, and flattening the output for final linear classification.

```text
Input Tensor: (Batch Size, 1, 28, 28)
 │
 ├── [Layer 1] Conv2d (in=1, out=8, kernel=3, padding=1)  ──> (Batch, 8, 28, 28)
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
 └── [Classifier 2] Linear (in=64, out=10) ──────────────────> (Batch, 10) [Logits Output]
```

### Layer Configuration Summary

| Layer Type | Input Shape | Output Shape | Parameters / Details |
| :--- | :--- | :--- | :--- |
| **Input Image** | `(1, 28, 28)` | `(1, 28, 28)` | Grayscale normalized tensor |
| **Conv2d (1)** | `(1, 28, 28)` | `(8, 28, 28)` | 8 filters, Kernel: `3x3`, Padding: `1` |
| **ReLU + MaxPool2d** | `(8, 28, 28)` | `(8, 14, 14)` | Kernel size: `2x2`, Stride: `2` |
| **Conv2d (2)** | `(8, 14, 14)` | `(16, 14, 14)` | 16 filters, Kernel: `3x3`, Padding: `1` |
| **ReLU + MaxPool2d** | `(16, 14, 14)` | `(16, 7, 7)` | Kernel size: `2x2`, Stride: `2` |
| **Flatten** | `(16, 7, 7)` | `(784)` | Linear feature unfolding |
| **Linear (Dense 1)** | `(784)` | `(64)` | Dense layer with ReLU activation |
| **Linear (Dense 2)** | `(64)` | `(10)` | 10-Class class output logits |

---

## 📊 Training Logs & Performance Report

The following report shows complete epoch loss values and final evaluation metrics across the full training cycle:

```python
Imported all Modules Successfully!
Device got choosen Successfully: cuda

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

## 🚀 Training & Inference Guide

### 1. Run Complete Training Pipeline
Run the main script to automatically download MNIST, train the model for 50 epochs, print the loss updates, test classification accuracy, and save weights to `model.pth`:

```bash
python train.py
```

### 2. Load Model for Testing / Custom Predictions
Load the model weights from the saved checkpoint file for testing or single-image inference:

```python
import torch
from train import SimpleCNN

# Select device (GPU / CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model instance and load trained weights
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

print("Model checkpoint loaded successfully and set to evaluation mode!")
```
