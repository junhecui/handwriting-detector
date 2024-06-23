import os
import xml.etree.ElementTree as ET
import cv2

def create_xml_annotation(image_path, label, output_dir):
    image = cv2.imread(image_path)

    height, width, channels = image.shape

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = os.path.basename(output_dir)
    ET.SubElement(annotation, 'filename').text = os.path.basename(image_path)
    ET.SubElement(annotation, 'path').text = image_path

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(channels)

    ET.SubElement(annotation, 'segmented').text = '0'

    obj = ET.SubElement(annotation, 'object')
    ET.SubElement(obj, 'name').text = label
    ET.SubElement(obj, 'pose').text = 'Unspecified'
    ET.SubElement(obj, 'truncated').text = '0'
    ET.SubElement(obj, 'difficult').text = '0'
    bndbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = '0'
    ET.SubElement(bndbox, 'ymin').text = '0'
    ET.SubElement(bndbox, 'xmax').text = str(width)
    ET.SubElement(bndbox, 'ymax').text = str(height)

    tree = ET.ElementTree(annotation)
    xml_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.xml')
    tree.write(xml_path)

def generate_annotations(base_dir, annotations_file):
    annotations = {}
    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                image_file = parts[0].strip()
                label = parts[1].strip()
                annotations[image_file] = label

    for image_file in os.listdir(base_dir):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(base_dir, image_file)
            xml_path = os.path.join(base_dir, os.path.splitext(image_file)[0] + '.xml')
            
            label = annotations.get(image_file)
            if label and not os.path.exists(xml_path):
                create_xml_annotation(image_path, label, base_dir)

base_dir = './segmented/segmented_speeches'
annotations_file = os.path.join(base_dir, 'annotations.txt')
generate_annotations(base_dir, annotations_file)