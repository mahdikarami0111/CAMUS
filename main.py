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
from models.TransUnet.train import train
from torch.utils.data import random_split


if __name__ == '__main__':
    cfg = TransUnet_cfg.get_train_config()
    train(cfg)









