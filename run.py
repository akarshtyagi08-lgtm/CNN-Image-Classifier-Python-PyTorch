import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from train import SimpleCNN

# 1. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Re-create model structure & load saved weights from disk
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))

# 3. Set to evaluation mode (turns off dropout/batchnorm training behavior)
model.eval()
print("Loaded pre-trained weights from model.pth successfully!\n")

# 4. Load test dataset (no training data needed)
transform = transforms.ToTensor()
test_data = MNIST(root="data", train=False, download=True, transform=transform)

# 5. Predict on test images without computing gradients
with torch.no_grad():
    for index in range(5):
        img_tensor, actual_label = test_data[index]
        
        # Reshape (1, 28, 28) -> (1, 1, 28, 28) for batch input
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get raw logits and pick highest score
        logits = model(img_tensor)
        predicted = logits.argmax(dim=1).item()
        
        print(f"Sample #{index + 1} -> Model Predicted: {predicted} | Real Label: {actual_label}")
