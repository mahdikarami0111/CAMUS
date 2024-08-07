# Based
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import skimage
from tqdm import tqdm
import torch.utils.data
from torch.utils.data import DataLoader
from utils.image import*
from models.Unet import Unet
from preprocess.preprocessor import*
import yaml
from train import select_transform
from torch.optim.lr_scheduler import ExponentialLR
from eval.evaluation import*
from utils import save
from dataset import CAMUS
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from utils.debug import*
from train import train
from dataset import Wrapper
from train import train_K_fold
from dotmap import DotMap
from skimage.metrics import hausdorff_distance
from models.TransUnet.TransUnet import VisionTransformer
from config import TransUnet_cfg
from train import train

from config.Unet_cfg import get_config
from torch.utils.data import random_split
from torch.utils.data import Subset

if __name__ == '__main__':
    # model = save.load_model("TransUnet", "transunetV2").to('cuda')
    # dataset = CAMUS({
    #     "root": "data/database_expanded",
    #     "device": "cuda",
    #     "type": "N 4CH",
    # })
    # indices = save.load_indices()
    # test = Subset(dataset, indices["test"])
    # test_set = Wrapper(test, transform=select_transform('basic'))
    #
    # test_loader = DataLoader(test_set, batch_size=8, shuffle=True, pin_memory=True)
    # print(calculate_dice_metric(model, test_loader, 'cuda'))
    train(get_config(), save.load_indices())









