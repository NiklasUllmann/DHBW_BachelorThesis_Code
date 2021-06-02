import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from PIL import Image

BATCH_SIZE = 64
SPLIT = 0.15
ANNOTATION_PATH = "./data/noisy_imagenette.csv"
IMG_PATH = "./data"
TRANSFORMER = transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ]
    )

class ImagenetteDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        transform,
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        # image = read_image(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        label = self.get_string_for_label(self.img_labels.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)
        return image, label

    def get_string_for_label(self, string):

        imagenette_map = {
            "n01440764": "tench",
            "n02102040": "springer",
            "n02979186": "casette_player",
            "n03000684": "chain_saw",
            "n03028079": "church",
            "n03394916": "French_horn",
            "n03417042": "garbage_truck",
            "n03425413": "gas_pump",
            "n03445777": "golf_ball",
            "n03888257": "parachute",
        }
        return imagenette_map[string]



def load_data():
    
    data = ImagenetteDataset(
        ANNOTATION_PATH, IMG_PATH, transform=TRANSFORMER
    )

    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(SPLIT * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, 
                                           sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE,
                                                sampler=val_sampler)
                                                
    return train_loader, validation_loader
