import cv2
import os
import numpy as np

def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 15, 4)
    
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    
    cv2.imwrite(output_path, binary_image)
    print(f"Processed image saved to {output_path}")
    
    return binary_image

image_path = './images/sentence-box.jpg'
processed_image_path = './processed/processed_sentences/sentence-box.jpg'
os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)

binary_image = preprocess_image(image_path, processed_image_path)
