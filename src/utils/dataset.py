import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class DatasetFromArray(Dataset):
    def __init__(self, images_array, labels_array, transform=None):
        self.images = images_array
        self.labels = labels_array
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long).squeeze()
        return img, label