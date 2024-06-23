import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CrossEntropyLoss  # Adjust according to your problem (e.g., CrossEntropyLoss for classification)
from torchvision import transforms

from datasets import HandwritingDataset
from model import HandwritingModel

# Define any transformations you want to apply to the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Define the base directory and categories
base_dir = './data/final'
categories = ['characters', 'sentences', 'speeches', 'words']

# Create an instance of the dataset and DataLoader
dataset = HandwritingDataset(base_dir=base_dir, categories=categories, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the model with the correct number of classes
num_classes = dataset.get_num_classes()
model = HandwritingModel(num_classes=num_classes)

# Define the loss function and the optimizer
criterion = CrossEntropyLoss()  # Adjust according to your problem
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Number of times to iterate over the entire dataset

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in data_loader:
        # Forward pass: Compute predicted outputs by passing inputs to the model
        outputs = model(images)
        loss = criterion(outputs, labels)  # Calculate the loss

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear the gradients of all optimized tensors
        loss.backward()  # Compute the gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}")

print("Finished Training")