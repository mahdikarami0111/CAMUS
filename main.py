# Based
import copy
import math

import PIL.Image
import albumentations
import pywt
from dataset import CAMUSP
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import skimage
from tqdm import tqdm
import torch.utils.data
from torch.utils.data import DataLoader
from utils.image import *
from models.Unet import Unet
from preprocess.preprocessor import *
import yaml
from train import select_transform
from torch.optim.lr_scheduler import ExponentialLR
from eval.evaluation import *
from utils import save
from dataset import CAMUS
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from utils.debug import *
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
from models.BCUnet.Convnext import LayerNorm
from models.BCUnet.BCUnet import BCUnet
from config.BCUnet_cfg import get_train_cfg
from models.BCUnet.train import train
from  models.DUCKnet.train import train as train_ducknet
from config.DuckUnet_cfg import get_DuckNet_train_config
from models.FCT.FCT import FCT
from models.FCT.train import train as train_fct
from config.FCT_cfg import get_FCT_config
from models.TransAttUnet.transformer_parts import PositionEmbeddingLearned
from models.TransAttUnet.TransAttUnet import TransAttUnet
from models.TransAttUnet.train import train as train_trans_att_unet
from config.TransAttUnet_cfg import get_TransAttUnet_train_config
from utils.wavelet import IDWT_2D, DWT_2D
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from models.WSegNet.WSegnet import WSegNetVGG, make_w_layers
from models.WSegNet.WSegnetv2 import WUNet
from models.train_unet import train as train_unet
from config.Unet_cfg import get_config as get_unet_config
from preprocess.preprocessor import copy_configs
import random
from models.WSegNet.WSegnetv2 import WUNet
from models.WSegNet.train import train as train_WaveUnet
from config.WaveUnet import get_config as get_WaveUnet_config



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
    # train(get_config(), save.load_indices())
    # _____________________________________________

    # dataset = CAMUS({
    #     "root": "data/database_expanded",
    #     "device": "cuda",
    #     "type": "N 4CH",
    # })
    # dataset = Wrapper(dataset, transform=select_transform("basic"))
    # model = BCUnet(1, 2).to('cuda')
    # loader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True)
    # for data in loader:
    #     X, Y = data
    #     out1, out2, out = model(X.to('cuda'))
    #     print(out.shape)
    #     break
    # __________________________________________________

    # model = save.load_model("BCUnet", "BCUnet").cuda()
    # indices = save.load_indices()
    # dataset = CAMUS({
    #     "root": "data/database_expanded",
    #     "device": "cuda",
    #     "type": "N 4CH",
    # })
    # test_set = Subset(dataset, indices["test"])
    # test_set = Wrapper(test_set, select_transform("basic"))
    # test_loader = DataLoader(test_set, batch_size=2, shuffle=True, pin_memory=True)
    # print(BCUnet_dice(model, test_loader, "cuda"))
    # __________________________________________________

    # model = save.load_model("ducknet", "DuckNet").to('cuda')
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
    # ___________________________________________________

    # train_trans_att_unet(get_TransAttUnet_train_config(), save.load_indices("data"))
    # ____________________________________________________

    # Wavelet
    # dataset = CAMUS({
    #     "root": "data/database_expanded",
    #     "device": "cuda",
    #     "type": "N 4CH",
    # })
    # img, mask = dataset[9614]
    # # show_tensor_img(img, mask)
    # img1 = np.stack((img,)*3, axis=-1).astype(np.uint8)
    # img = torch.from_numpy(img1).float()
    #
    #
    # img = img.permute(2, 0, 1)
    # img = img.unsqueeze(0).cuda()
    # #
    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 10), sharex=True, sharey=True)
    # plt.gray()
    # dwt = DWT_2D(in_channels=3)
    # ll, lh, hl, hh = dwt(img)
    # ll = ll.squeeze(0).permute(1, 2, 0).cpu()
    # ll = (ll - torch.min(ll)) / (torch.max(ll) - torch.min(ll))
    # ll = np.array(ll)
    #
    # lh = lh.squeeze(0).permute(1, 2, 0).cpu()
    # t = torch.min(lh)
    # m = torch.max(lh)
    # lh = (lh - t) / (m - t)
    # lh = np.array(lh)
    #
    # hl = hl.squeeze(0).permute(1, 2, 0).cpu()
    # t = (torch.min(hl))
    # m = torch.max(hl)
    # hl = torch.nn.functional.softshrink(hl, 5)
    # hl = (hl - t) / (m - t)
    # hl = np.array(hl)
    #
    # hh = hh.squeeze(0).permute(1, 2, 0).cpu()
    # t = (torch.min(hh))
    # m = torch.max(hh)
    # hh = (hh - t) / (m - t)
    # hh = np.array(hh)
    #
    # print(ll.shape)
    # ax[0,0].imshow(ll)
    # ax[0,0].axis('off')
    # ax[0,0].set_title('ll')
    #
    # ax[1,0].imshow(hl)
    # ax[1,0].axis('off')
    # ax[1,0].set_title('hl')
    #
    # ax[0, 1].imshow(lh)
    # ax[0, 1].axis('off')
    # ax[0, 1].set_title('lh')
    #
    # ax[1, 1].imshow(hh)
    # ax[1, 1].axis('off')
    # ax[1, 1].set_title('hh')
    # plt.show()

    # train_unet(get_unet_config(), save.load_indices("data"))

    # dir = "data/database_expanded"
    # list = os.listdir(dir)
    # list.sort()
    #
    # good = []
    # medium = []
    # poor = []
    #
    # for i, subject in enumerate(list):
    #
    #     path = f"{dir}/{subject}"
    #     with open(f"{path}/Info_4CH.cfg", 'r') as file:
    #         for line in file:
    #             key, value = line.strip().split(': ')
    #             if key != "ImageQuality":
    #                 continue
    #             if value == "Good":
    #                 good.append(i)
    #             elif value == "Medium":
    #                 medium.append(i)
    #             elif value == "Poor":
    #                 poor.append(i)
    # random.shuffle(good)
    # random.shuffle(medium)
    # random.shuffle(poor)
    #
    # train = []
    # test = []
    # val = []
    #
    # cur = good
    # l = len(cur)
    # tl = math.floor(l * 0.75)
    # vl = math.floor(l * 0.1)
    # train += cur[0:tl]
    # val += cur[tl:tl+vl]
    # test += cur[tl+vl:]
    #
    # cur = medium
    # l = len(cur)
    # tl = math.floor(l * 0.75)
    # vl = math.ceil(l * 0.1)
    # train += cur[0:tl]
    # val += cur[tl:tl + vl]
    # test += cur[tl + vl:]
    #
    # cur = poor
    # l = len(cur)
    # tl = math.floor(l * 0.75)
    # vl = math.ceil(l * 0.1)
    # train += cur[0:tl]
    # val += cur[tl:tl + vl]
    # test += cur[tl + vl:]
    #
    # test = np.array(test)
    # val = np.array(val)
    # train = np.array(train)
    #
    # temp = save.load_QP_indices()
    # print(temp["test"])
    # ________________________________________________

    train_WaveUnet(get_WaveUnet_config(), save.load_QP_indices())





