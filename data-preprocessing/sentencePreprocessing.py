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

def segment_sentences(binary_image, output_dir, min_area=1000):
    horizontal_projection = np.sum(binary_image, axis=1)
    
    threshold = np.mean(horizontal_projection) * 0.5
    line_positions = np.where(horizontal_projection > threshold)[0]

    line_boundaries = []
    start = line_positions[0]
    for i in range(1, len(line_positions)):
        if line_positions[i] != line_positions[i - 1] + 1:
            end = line_positions[i - 1]
            line_boundaries.append((start, end))
            start = line_positions[i]
    line_boundaries.append((start, line_positions[-1]))

    sentence_index = 0
    for start, end in line_boundaries:
        sentence_image = binary_image[start:end, :]
        padded_sentence_image = pad_image(sentence_image, sentence_image.shape[0], binary_image.shape[1])
        
        sentence_image_path = os.path.join(output_dir, f'sentence_{sentence_index}.png')
        cv2.imwrite(sentence_image_path, padded_sentence_image)
        sentence_index += 1

def pad_image(image, target_height, target_width):
    height, width = image.shape
    padded_image = np.zeros((target_height, target_width), dtype=np.uint8)
    y_offset = (target_height - height) // 2
    x_offset = (target_width - width) // 2
    padded_image[y_offset:y_offset+height, x_offset:x_offset+width] = image
    return padded_image

image_path = './images/sentence-fox.jpg'
processed_image_path = './processed/processed_sentences/sentence-fox.jpg'
output_dir = './temp/segmented_sentences'
os.makedirs(output_dir, exist_ok=True)

binary_image = preprocess_image(image_path, processed_image_path)

if binary_image is not None:
    segment_sentences(binary_image, output_dir, min_area=1000)