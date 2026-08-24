# ✍️ MNIST Handwritten Digit Recognition (PyTorch CNN)

A Convolutional Neural Network (CNN) built using **PyTorch** to classify handwritten digits (0–9) on the **MNIST** dataset[span_0](start_span)[span_0](end_span). The model reaches a final test accuracy of **98.69%** over 50 epochs[span_1](start_span)[span_1](end_span).

---

## 📌 Project Overview

* **Framework:** PyTorch & Torchvision[span_2](start_span)[span_2](end_span)
* **Dataset:** MNIST (28x28 grayscale handwritten digit images)[span_3](start_span)[span_3](end_span)
* **Architecture:** 2 Convolutional layers + MaxPooling + 2 Linear layers[span_4](start_span)[span_4](end_span)
* **Loss Function:** CrossEntropyLoss[span_5](start_span)[span_5](end_span)
* **Optimizer:** Adam (lr = 0.001)[span_6](start_span)[span_6](end_span)
* **Batch Size:** 64[span_7](start_span)[span_7](end_span)
* **Epochs:** 50[span_8](start_span)[span_8](end_span)
* **Final Test Accuracy:** **98.69%**

---

## 🧠 Model Architecture

```text
Input (1x28x28)
│
├── Conv2d (1 → 8 filters, kernel=3, padding=1) + ReLU
├── MaxPool2d (kernel=2) → Output: (8x14x14)
│
├── Conv2d (8 → 16 filters, kernel=3, padding=1) + ReLU
├── MaxPool2d (kernel=2) → Output: (16x7x7)
│
├── Flatten (16 * 7 * 7 = 784)
├── Linear (784 → 64) + ReLU
└── Linear (64 → 10) → Output

📊 Training Logs & Performance
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

🛠️ Installation & Requirements
Install the dependencies:
pip install torch torchvision

🚀 Execution & Usage
1. Training the Model
Run the script to download the MNIST dataset, train across 50 epochs, display results, and export model.pth:
python train.py

2. Performing Inference
Load the trained weights model.pth directly into the network:
import torch
from train import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
print("Model loaded successfully!")


