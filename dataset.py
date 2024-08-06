import glob
import os

import nibabel as nib
import torchvision.transforms as torch_transforms
import numpy as np
from torch.utils import data
import albumentations as A
from utils.image import*
from PIL import Image
from albumentations.pytorch import ToTensorV2


class CAMUS(data.Dataset):
    def __init__(self, dataset_cfg: dict):
        self.root = dataset_cfg["root"]
        self.device = dataset_cfg["device"]
        self.type = dataset_cfg["type"]
        self.data = self.get_data()
        self.basic_transform = torch_transforms.ToTensor()
        self.normalize = torch_transforms.Normalize([0.5], [0.5])

    def get_data(self):
        data_ = []
        dataset_dir = self.root
        subjects = os.listdir(dataset_dir)
        subjects.sort()
        for subject in subjects:
            images = sorted(glob.glob(f"{dataset_dir}/{subject}/*_.jpg"))
            masks = sorted(glob.glob(f"{dataset_dir}/{subject}/*_gt.png"))
            for i in range(len(images)):
                if images[i][-10:].split("_")[1] != masks[i][-12:].split("_")[1]:
                    print("Error image and mask not matching")
                    return None
                data_.append((images[i], masks[i]))
        return data_

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, mask = self.data[item]
        img = np.array(Image.open(img), dtype=np.float32)
        mask = np.array(Image.open(mask), dtype=np.float32)
        mask = mask // 85
        mask[mask != 1] = 0
        return img, mask


class Wrapper(data.Dataset):
    def __init__(self, subset, transform=None):
        self.normalize = torch_transforms.Normalize([0.5], [0.5])
        self.subset = subset
        self.transform = transform

    def __getitem__(self, item):
        img, mask = self.subset[item]
        img_tensor = self.transform(image=img, mask=mask)
        img = self.normalize(img_tensor['image'])
        mask = img_tensor['mask']
        mask = mask.unsqueeze(0)
        return img, mask

    def __len__(self):
        return len(self.subset)






