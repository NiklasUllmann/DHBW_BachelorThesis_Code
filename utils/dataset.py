import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms.functional as TF
import torch
from PIL import Image
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomVerticalFlip

BATCH_SIZE = 4
NUM_WORKERS = 6
SPLIT = 0.3
ANNOTATION_PATH = "./data/noisy_imagenette_extended.csv"
IMG_PATH = "./data"
TRANSFORMER = transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

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
            "n01440764": 0,
            "n02102040": 1,
            "n02979186": 2,
            "n03000684": 3,
            "n03028079": 4,
            "n03394916": 5,
            "n03417042": 6,
            "n03425413": 7,
            "n03445777": 8,
            "n03888257": 9,
        }
        return imagenette_map[string]


def load_data(batch_size):

    data = ImagenetteDataset(
        ANNOTATION_PATH, IMG_PATH, transform=TRANSFORMER
    )

    dataset_size = len(data)
    print(dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(SPLIT * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                    sampler=val_sampler, num_workers=NUM_WORKERS, pin_memory=True)
                                                
    return train_loader, validation_loader


def load_single_image(path):

    image = Image.open(path)

    # Create SingleTensor
    x = TF.resize(image, [320, 320])
    x = TF.to_tensor(x)
    x = TF.normalize(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
    #Create PilImg
    y = TF.to_pil_image(x)

    x.unsqueeze_(0)
    return x, y

def just_load_resize_pil(path):
    image = Image.open(path)
    x = TF.resize(image, [320, 320])
    x = TF.to_tensor(x)
    x = TF.to_pil_image(x)
    return x


