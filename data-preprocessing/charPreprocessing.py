import cv2
import os
import numpy as np

def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    
    cv2.imwrite(output_path, binary_image)
    print(f"Processed image saved to {output_path}")
    
    return binary_image

def segment_characters(binary_image, output_dir, min_area=50):
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_index = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        char_image = binary_image[y:y+h, x:x+w]
        char_image = cv2.resize(char_image, (28, 28))
        
        char_image_path = os.path.join(output_dir, f'char_{char_index}.png')
        cv2.imwrite(char_image_path, char_image)
        char_index += 1

image_path = './images/characters-4-9.jpg'
processed_image_path = './processed_characters/processed_characters-4-9.jpg'
output_dir = './temp/segmented_characters'
os.makedirs(output_dir, exist_ok=True)

binary_image = preprocess_image(image_path, processed_image_path)

if binary_image is not None:
    segment_characters(binary_image, output_dir, min_area=50)