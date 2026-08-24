# ✍️ MNIST Handwritten Digit Recognition (PyTorch CNN)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Accuracy-98.69%25-success?style=for-the-badge" alt="Accuracy" />
  <img src="https://img.shields.io/badge/Dataset-MNIST-blue?style=for-the-badge" alt="MNIST" />
</p>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" width="450" alt="MNIST Handwritten Digits Sample" />
</p>

A deep learning project implementing a Convolutional Neural Network (CNN) in **PyTorch** to classify handwritten digits (0 through 9) using the benchmark **MNIST** dataset. The model achieves **98.69% test accuracy** after 50 epochs of training.

---

## 📌 Project Highlights

* 🖼️ **Dataset:** MNIST Grayscale Handwritten Digits (28x28)
* 🧠 **Architecture:** 2 Convolutional Layers + Max Pooling + 2 Fully Connected Layers
* ⚙️ **Optimizer:** Adam (Learning Rate = 0.001)
* 📉 **Loss Function:** CrossEntropyLoss
* 📦 **Batch Size:** 64
* 🔁 **Epochs:** 50
* 🎯 **Final Test Accuracy:** 98.69%

---

## 📦 Required Libraries & Dependencies

Make sure you have the following packages installed in your environment:

* `python` (>= 3.8)
* `torch` (PyTorch core library for deep learning operations)
* `torchvision` (For downloading and transforming the MNIST dataset)

To install all required libraries at once, run:

pip install torch torchvision

---

## 🧠 Model Architecture

The custom `SimpleCNN` network processes 28x28 single-channel images through the following pipeline:

Input (1x28x28)
  │
  ├── Conv2d (in_channels=1, out_channels=8, kernel_size=3, padding=1) + ReLU
  ├── MaxPool2d (kernel_size=2)  ──> Output shape: (8, 14, 14)
  │
  ├── Conv2d (in_channels=8, out_channels=16, kernel_size=3, padding=1) + ReLU
  ├── MaxPool2d (kernel_size=2)  ──> Output shape: (16, 7, 7)
  │
  ├── Flatten (16 * 7 * 7 = 784 elements)
  ├── Linear (784 ──> 64) + ReLU
  └── Linear (64 ──> 10)  ──> Logits for digits 0 to 9

---

## 📊 Training Logs & Performance

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

---

## 🚀 How to Run

### 1. Training & Evaluation
Run the training script to fetch the MNIST dataset, train across 50 epochs, print test accuracy, and save weights to `model.pth`:

python train.py

### 2. Loading the Model for Predictions

import torch
from train import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

print("Trained model loaded successfully!")
