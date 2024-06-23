import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

class HandwritingDataset(Dataset):
    def __init__(self, base_dir, categories, transform=None):
        self.img_labels = []
        self.base_dir = base_dir
        self.transform = transform
        self.categories = categories
        self.label_to_index = {}
        self.index_to_label = {}
        self._load_labels()

        for category in categories:
            category_dir = os.path.join(base_dir, category)
            if category == 'speeches':
                annotations_file = os.path.join(category_dir, 'annotations.txt')
                self._load_speeches(annotations_file, category_dir)
            else:
                self._load_category(category_dir)

    def _load_labels(self):
        index = 0
        for category in self.categories:
            category_dir = os.path.join(self.base_dir, category)
            if category == 'speeches':
                annotations_file = os.path.join(category_dir, 'annotations.txt')
                with open(annotations_file, 'r') as file:
                    for line in file:
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) == 2:
                            label = parts[1].strip()
                            if label not in self.label_to_index:
                                self.label_to_index[label] = index
                                self.index_to_label[index] = label
                                index += 1
            else:
                for class_dir in os.listdir(category_dir):
                    class_path = os.path.join(category_dir, class_dir)
                    if os.path.isdir(class_path):
                        for img_file in os.listdir(class_path):
                            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                                xml_path = os.path.splitext(os.path.join(class_path, img_file))[0] + '.xml'
                                if os.path.exists(xml_path):
                                    label = self._parse_xml(xml_path)
                                    if label not in self.label_to_index:
                                        self.label_to_index[label] = index
                                        self.index_to_label[index] = label
                                        index += 1

    def _load_speeches(self, annotations_file, category_dir):
        with open(annotations_file, 'r') as file:
            for line in file:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    img_path = os.path.join(category_dir, parts[0].strip())
                    label = parts[1].strip()
                    self.img_labels.append((img_path, self.label_to_index[label]))

    def _load_category(self, category_dir):
        for class_dir in os.listdir(category_dir):
            class_path = os.path.join(category_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.endswith('.jpg') or img_file.endswith('.png'):
                        img_path = os.path.join(class_path, img_file)
                        xml_path = os.path.splitext(img_path)[0] + '.xml'
                        if os.path.exists(xml_path):
                            label = self._parse_xml(xml_path)
                            self.img_labels.append((img_path, self.label_to_index[label]))

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        label = root.find('object/name').text
        return label

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels[idx][0]
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels[idx][1]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label)

        return image, label
    
    def get_num_classes(self):
        return len(self.label_to_index)