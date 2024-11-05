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

    def bayes_thresh(lh, hl, hh):
        channels = lh.shape[1]
        batch_size = lh.shape[0]
        height = hh.shape[2]
        width = hh.shape[3]
        sigma = torch.median(torch.abs(hh.view(batch_size, channels, height * width)), dim=2)[0] / .6745

        sigma_y = torch.stack((lh, hl, hh), dim=1).view(batch_size, 3, channels, height * width)
        sigma_y = torch.mean(torch.pow(sigma_y, 2), dim=-1)

        sigma = sigma.unsqueeze(dim=1)
        sigma_x = torch.sqrt(torch.maximum(sigma_y - torch.pow(sigma, 2), torch.tensor(0.0001)))
        T = torch.pow(sigma, 2) / sigma_x
        return T

    def denoise_wavelet(lh, hl, hh, T):
        channels = lh.shape[1]
        batch_size = lh.shape[0]
        height = hh.shape[2]
        width = hh.shape[3]
        coeffs = torch.stack((lh, hl, hh), dim=1).view(batch_size, 3, channels, height * width)
        m = torch.abs(coeffs)
        T = T.unsqueeze(-1)
        denom = m + (m < T).float()
        gain = torch.maximum(m-T, torch.tensor(0))/denom
        coeffs = (coeffs * gain).view(batch_size, 3, channels, height, width)
        lh = coeffs[:, 0, :, :, :].squeeze(1)
        hl = coeffs[:, 1, :, :, :].squeeze(1)
        hh = coeffs[:, 2, :, :, :].squeeze(1)

        return lh, hl, hh







    dataset = CAMUS({
        "root": "data/database_expanded",
        "device": "cuda",
        "type": "N 4CH",
    })
    dataset = Wrapper(dataset, transform=select_transform('basic'))
    img, mask = dataset[9614]
    img2, mask2 = dataset[2263]
    img3, mask3 = dataset[4562]
    img4, mask4 = dataset[6732]
    # show_tensor_img(img, mask)

    img = torch.stack((img, img2, img3, img4)).cuda()
    img = torch.cat((img, img, img), dim=1).cuda()
    # print(img.shape)

    #
    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 10), sharex=True, sharey=True)
    # plt.gray()
    dwt = DWT_2D(in_channels=3)
    ll, lh, hl, hh = dwt(img)
    T = bayes_thresh(lh, hl, hh)
    lh_, hl, hh = denoise_wavelet(lh, hl, hh, T)



    npimg = np.asarray(img2)
    npimg = npimg.transpose((1, 2, 0))
    npimg = np.concatenate((npimg,)*3, axis=-1).astype(np.uint8)
    coeffs = pywt.wavedec2(npimg, 'haar', axes=(0, 1), level=1)

    temp = torch.cat((img2.unsqueeze(0).cuda(),)*3, dim=1)
    ll, lh, hl, hh = dwt(temp)
    ll = ll.squeeze(0).permute(1, 2, 0).cpu()
    ll = np.array(ll)
    lh = lh.squeeze(0).permute(1, 2, 0).cpu()
    lh = np.array(lh)
    hl = hl.squeeze(0).permute(1, 2, 0).cpu()
    hl = np.array(hl)
    hh = hh.squeeze(0).permute(1, 2, 0).cpu()
    hh = np.array(hh)
    coeffs = [ll, [lh, hl, hh]]

    bandpasses = coeffs[1:]
    C = 3
    σ = np.zeros((C))
    for c in range(C):
        σ[c] = np.median(np.abs(bandpasses[0][2][:, :, c].ravel())) / .6745
    σ = σ.reshape(1, 1, C)

    # Estimate the variance of the noisy signal for each subband, scale and channel
    σy2 = np.zeros((1, 3, C))
    for j in range(1):
        for b in (0, 1, 2):
            for c in range(C):
                σy2[j, b, c] = np.mean(bandpasses[j][b][:, :, c] ** 2)

    # Calculate σ_x = sqrt(σ_y^2 - σ^2) for each subband, scale and channel
    σx = np.sqrt(np.maximum(σy2 - σ ** 2, 0.0001))

    # Calculate T
    T = (σ ** 2) / σx


    def shrink(x: np.ndarray, t: float):
        """ Given a wavelet coefficient and a threshold, shrink """
        if t == 0:
            return x
        m = np.abs(x)
        denom = m + (m < t).astype('float')
        gain = np.maximum(m - t, 0) / denom
        return x * gain


    def shrink_coeffs(coeffs: np.ndarray, T: np.ndarray):
        """ Shrink the wavelet coefficients with the thresholds T.

        coeffs should be the output of pywt.wavedec (list of numpy arrays)
        T should be an array of shape (J, 3, C) for color images.
        """
        assert T.shape[0] == len(coeffs) - 1
        J = len(coeffs) - 1
        assert T.shape[1] == len(coeffs[1])
        assert T.shape[2] == len(coeffs[1][0][0, 0])
        C = T.shape[2]

        coeffs_new = [None, ] * (J + 1)
        coeffs_new[0] = np.copy(coeffs[0])
        for j in range(J):
            coeffs_new[1 + j] = [np.zeros_like(coeffs[1 + j][0]),
                                 np.zeros_like(coeffs[1 + j][1]),
                                 np.zeros_like(coeffs[1 + j][2])]
            for b, band in enumerate(['LH', 'HL', 'HH']):
                for c in range(C):
                    temp = shrink(coeffs[1 + j][b][:, :, c], T[j, b, c])
                    coeffs_new[1 + j][b][:, :, c] = temp

        return coeffs_new


    l = shrink_coeffs(coeffs, T)
    lh = l[1][0]
    lh_ = lh_[1, :, :, :].squeeze(0).permute(1, 2, 0).cpu()
    lh_ = np.array(lh_)
    print(np.max(lh_))
    print(np.min(lh_))
    distance = lh - lh_
    print(np.max(distance))
    print(np.min(distance))




    # print(lh.shape)
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









