from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import torch
from model.model import SimpleCNN
from PIL import Image
import torchvision.transforms as transforms
from preprocessing.speechPreprocessing import preprocess_image

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

num_classes = 62 
model = SimpleCNN(num_classes=num_classes)
state_dict = torch.load('character_recognition_model.pth')
model.load_state_dict(state_dict)
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.route('/')
def upload_form():
    return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Upload Image</title>
        </head>
        <body>
            <h1>Upload an image of your handwriting</h1>
            <form action="/upload" method="POST" enctype="multipart/form-data">
                <input type="file" name="file"><br><br>
                <input type="submit" value="Upload Image">
            </form>
        </body>
        </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
        preprocess_image(file_path, processed_image_path)

        predicted_class = predict_image(processed_image_path)

        return f'Prediction: {predicted_class}'

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities, predicted_indices = torch.topk(outputs, 3)

    probabilities = torch.nn.functional.softmax(probabilities, dim=1) * 100

    index_to_label = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
        13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
        26: 'a', 27: 'b', 28: 'c', 29: 'd', 30: 'e', 31: 'f', 32: 'g', 33: 'h', 34: 'i', 35: 'j', 36: 'k', 37: 'l', 38: 'm',
        39: 'n', 40: 'o', 41: 'p', 42: 'q', 43: 'r', 44: 's', 45: 't', 46: 'u', 47: 'v', 48: 'w', 49: 'x', 50: 'y', 51: 'z',
        52: '0', 53: '1', 54: '2', 55: '3', 56: '4', 57: '5', 58: '6', 59: '7', 60: '8', 61: '9'
    }

    top_predictions = []
    for i in range(3):
        label = index_to_label[predicted_indices[0][i].item()] 
        probability = probabilities[0][i].item() 
        top_predictions.append(f'{label}: {probability:.2f}%')

    return top_predictions

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)
    app.run(debug=True)