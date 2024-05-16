import glob
import nibabel as nib
import numpy as np
import torchvision.transforms as torch_transforms
import torch
from tools.image import *
from torch.utils import data
import cv2
import albumentations as A


class CAMUS(data.Dataset):
    def __init__(self, dataset_cfg: dict, transform: A.core.composition.Compose = None):
        self.root = dataset_cfg["root"]
        self.device = dataset_cfg["device"]
        self.type = dataset_cfg["type"]
        self.transform = transform
        self.data, self.labels = self.get_data()
        self.basic_transform = torch_transforms.ToTensor()

    def get_data(self):
        dataset_dir = self.root + "/database"
        print(dataset_dir)
        img_list = glob.glob(f"{dataset_dir}/*/*4CH_ED.nii")
        img_list.extend(glob.glob(f"{dataset_dir}/*/*4CH_ES.nii"))
        mask_list = glob.glob(f"{dataset_dir}/*/*4CH_ED_gt.nii")
        mask_list.extend(glob.glob(f"{dataset_dir}/*/*4CH_ES_gt.nii"))

        return img_list, mask_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        print(self.data[item])
        img = nib.load(self.data[item])
        img = np.array(np.transpose(img.get_fdata()), dtype=np.uint8)
        mask = nib.load(self.labels[item])
        mask = np.array(np.transpose(mask.get_fdata()), dtype=np.uint8)
        mask = np.where(mask == 1, mask, 0)
        img_tensor = self.transform(image=img, mask=mask)
        #show_tensor_img(img_tensor['image'], img_tensor['mask'])
        return img_tensor['image'], img_tensor['mask']







