import glob
import nibabel as nib
import torchvision.transforms as torch_transforms
import numpy as np
from torch.utils import data
import albumentations as A
from utils.image import*
from albumentations.pytorch import ToTensorV2


class CAMUS(data.Dataset):
    def __init__(self, dataset_cfg: dict, transform: A.core.composition.Compose = None):
        self.root = dataset_cfg["root"]
        self.device = dataset_cfg["device"]
        self.type = dataset_cfg["type"]
        self.transform = transform
        self.data, self.labels = self.get_data()
        self.basic_transform = torch_transforms.ToTensor()
        self.normalize = torch_transforms.Normalize([0.5], [0.5])

    def get_data(self):
        dataset_dir = self.root + "/database"
        img_list = glob.glob(f"{dataset_dir}/*/*4CH_ED.nii")
        img_list.extend(glob.glob(f"{dataset_dir}/*/*4CH_ES.nii"))
        mask_list = glob.glob(f"{dataset_dir}/*/*4CH_ED_gt.nii")
        mask_list.extend(glob.glob(f"{dataset_dir}/*/*4CH_ES_gt.nii"))

        return img_list, mask_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = nib.load(self.data[item])
        img = np.array(np.transpose(img.get_fdata()), dtype=np.float32)
        mask = nib.load(self.labels[item])
        mask = np.array(np.transpose(mask.get_fdata()), dtype=np.float32)
        mask = np.where(mask == 1, mask, 0)
        img_tensor = self.transform(image=img, mask=mask)
        img = self.normalize(img_tensor['image'])
        mask = img_tensor['mask']
        mask = mask.unsqueeze(0)
        # show_tensor_img(img_tensor['image'], img_tensor['mask'])
        return img, mask







