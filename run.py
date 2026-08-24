import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from train import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize the model architecture and load trained weights
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

# 2. Load MNIST test dataset to run sample predictions
transform = transforms.ToTensor()
test_data = MNIST(root="data", train=False, download=True, transform=transform)

# 3. Test on the first 5 samples
print("\n--- Running Sample Predictions ---")
with torch.no_grad():
    for i in range(5):
        image, actual_label = test_data[i]
        
        # Add batch dimension: (1, 1, 28, 28)
        input_tensor = image.unsqueeze(0).to(device)
        
        # Forward pass
        output = model(input_tensor)
        predicted_label = output.argmax(dim=1).item()
        
        # Calculate confidence using softmax
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0][predicted_label].item() * 100
        
        print(f"Sample {i+1} -> Predicted: {predicted_label} | Actual: {actual_label} | Confidence: {confidence:.2f}%")
