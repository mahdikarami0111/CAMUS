import cv2
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import skimage
import torch.utils.data
from torchvision.models import 
from preprocess.preprocessor import*
import yaml
from dataset import CAMUS
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tools.debug import*

# img = nib.load("data/database/patient0001/patient0001_4CH_ES.nii")
# nii_data = np.array(np.transpose(img.get_fdata()), dtype=np.uint8)
#
# cv2.imshow("img", nii_data)
# cv2.waitKey(0)
# do_stuff("data/database/patient0001/patient0001_2CH_ES.nii", "data/database/patient0001/patient0001_2CH_ES_gt.nii")
with open("config/dataset_config.yaml", 'r') as cfg:
    temp = yaml.safe_load(cfg)

# crop_ratio = 1.0
input_size = 224
input_transformer = A.Compose([
            A.HorizontalFlip(p=0.5),
            # A.Resize(height=input_size, width=input_size),
            ToTensorV2(),
            ])

dataset = CAMUS(temp, input_transformer)
x = dataset[501]
train_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.3])





