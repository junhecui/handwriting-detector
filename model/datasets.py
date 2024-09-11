import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

class CharacterDataset(Dataset):
    def __init__(self, base_dir, label_to_index, transform=None):
        self.img_labels = []
        self.base_dir = base_dir
        self.transform = transform
        self.label_to_index = label_to_index
        self._load_data()

    def _load_data(self):
        for character_folder in os.listdir(self.base_dir):
            character_dir = os.path.join(self.base_dir, character_folder)
            if os.path.isdir(character_dir):
                for img_file in os.listdir(character_dir):
                    if img_file.endswith('.jpg') or img_file.endswith('.png'):
                        img_path = os.path.join(character_dir, img_file)
                        xml_path = os.path.splitext(img_path)[0] + '.xml'
                        if os.path.exists(xml_path):
                            label = self._parse_xml(xml_path)
                            label_index = self._convert_label_to_index(label)
                            if label_index != -1:
                                self.img_labels.append((img_path, label_index))

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        label = root.find('object/name').text
        return label

    def _convert_label_to_index(self, label):
        return self.label_to_index.get(label, -1)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)