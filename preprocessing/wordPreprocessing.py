import cv2
import os
import numpy as np

def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    
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

def pad_image(image, target_height, target_width):
    height, width = image.shape
    padded_image = np.zeros((target_height, target_width), dtype=np.uint8)
    y_offset = (target_height - height) // 2
    x_offset = (target_width - width) // 2
    padded_image[y_offset:y_offset+height, x_offset:x_offset+width] = image
    return padded_image

def segment_words(binary_image, output_dir, min_area=100):
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    word_index = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        word_image = binary_image[y:y+h, x:x+w]
        word_image = cv2.resize(word_image, (128, 48))
        
        word_image_path = os.path.join(output_dir, f'word_{word_index}.png')
        cv2.imwrite(word_image_path, word_image)
        word_index += 1

image_path = './images/words-wonderful-universe-fantastic-computer-keyboard-language-question.jpg'
processed_image_path = './processed/processed_words/words-wonderful-universe-fantastic-computer-keyboard-language-question.jpg'
output_dir = './temp/segmented_words'
os.makedirs(output_dir, exist_ok=True)

binary_image = preprocess_image(image_path, processed_image_path)

if binary_image is not None:
    segment_words(binary_image, output_dir, min_area=100)