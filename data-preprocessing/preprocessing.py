import cv2
import os

def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    
    cv2.imwrite(output_path, binary_image)
    print(f"Processed image saved to {output_path}")
    
    return binary_image

def segment_characters(binary_image, output_dir, min_area=100):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

image_path = './images/characters-A-G.jpg'
processed_image_path = './processed/processed_characters-A-G.jpg'
output_dir = './segmented/segmented_characters'
os.makedirs(output_dir, exist_ok=True)

binary_image = preprocess_image(image_path, processed_image_path)

if binary_image is not None:
    segment_characters(binary_image, output_dir, min_area=100)
