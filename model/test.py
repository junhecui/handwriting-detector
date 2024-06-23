import torch
from PIL import Image
import torchvision.transforms as transforms
from model import HandwritingModel

# Load the preprocessed image
def load_preprocessed_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Load the actual text
def load_actual_text(text_file_path):
    with open(text_file_path, 'r') as file:
        actual_text = file.read().strip()
    return actual_text

# Load the model
def load_model(model_path, num_classes):
    model = HandwritingModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Make prediction
def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Compare predictions with actual text
def compare_predictions(actual_text, predicted_class):
    print(f'Actual text: {actual_text}')
    print(f'Predicted class: {predicted_class}')
    if actual_text == predicted_class:
        print('Prediction is correct!')
    else:
        print('Prediction is incorrect.')

# Paths to the model, preprocessed image, and text file
model_path = './model/model.py'
preprocessed_image_path = './processed.jpg'
text_file_path = './text.txt'

# Number of classes
num_classes = 10  # Replace with the actual number of classes

# Load the preprocessed image
preprocessed_image = load_preprocessed_image(preprocessed_image_path)

# Load the actual text
actual_text = load_actual_text(text_file_path)

# Load the model
model = load_model(model_path, num_classes)

# Make prediction
predicted_label = predict(model, preprocessed_image)

# Map the label index to actual class
index_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
predicted_class = index_to_label[predicted_label]

# Compare predictions with actual text
compare_predictions(actual_text, predicted_class)
