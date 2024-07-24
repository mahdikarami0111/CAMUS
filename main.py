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
from torch.optim.lr_scheduler import ExponentialLR
from eval.evaluation import*
from utils import save
from dataset import CAMUS
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from utils.debug import*
from train import train
from train import train_K_fold
from dotmap import DotMap
from skimage.metrics import hausdorff_distance

# img = nib.load("data/database/patient0001/patient0001_4CH_ES.nii")
# nii_data = np.array(np.transpose(img.get_fdata()), dtype=np.uint8)
#
# cv2.imshow("img", nii_data)
# cv2.waitKey(0)
# do_stuff("data/database/patient0001/patient0001_2CH_ES.nii", "data/database/patient0001/patient0001_2CH_ES_gt.nii")
if __name__ == '__main__':
    # with open("config/Unet_00.yaml", 'r') as cfg:
    #     cfg = yaml.safe_load(cfg)
    # train_K_fold(DotMap(cfg))
    print(224 // 16 // 16)
    # crop_ratio = 1.0
    # input_size = 224
    # device = "cuda"
    # input_transformer = A.Compose([
    #     # A.HorizontalFlip(p=0.5),
    #     # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=1.0),
    #     A.Resize(height=input_size, width=input_size),
    #     ToTensorV2(),
    #     ])
    #
    # dataset = CAMUS(cfg, input_transformer)
    # train_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.3])
    #
    # train_loader = DataLoader(train_set, shuffle=True, batch_size=8, pin_memory=True)
    # test_loader = DataLoader(test_set, shuffle=True, batch_size=8, pin_memory=True)
    #
    # kf = KFold(10, shuffle=True)
    # for i, (train_index, test_index) in enumerate(kf.split(dataset)):
    #     print(f"Fold {i}:")
    #     print(f"  Train: index={len(train_index)}")
    #     print(f"  Test:  index={len(test_index)}")

    # unet = Unet(1, 1).to(device)
    # loss_function = torch.nn.BCEWithLogitsLoss()
    # opt = torch.optim.Adam(unet.parameters(), 0.02)
    # scheduler = ExponentialLR(opt, gamma=0.95)
    #
    # train_steps = len(train_set) // 8
    # test_steps = len(test_set) // 8
    #
    # h = {"train_loss": [], "test_loss": []}
    #
    # for e in tqdm(range(30)):
    #     unet.train()
    #
    #     total_train_loss = 0
    #     total_test_loss = 0
    #
    #     for(i, (X, Y)) in enumerate(train_loader):
    #         (X, Y) = (X.to(device), Y.to(device))
    #         out = unet(X)
    #         loss = loss_function(out, Y)
    #
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #
    #         total_train_loss += loss
    #
    #     with torch.no_grad():
    #         unet.eval()
    #
    #         for(x, y) in test_loader:
    #             (x, y) = (x.to(device), y.to(device))
    #             out = unet(x)
    #             total_test_loss += loss_function(out, y)
    #
    #     avg_train_loss = total_train_loss / train_steps
    #     avg_test_loss = total_test_loss / test_steps
    #     scheduler.step()
    #
    #     h["train_loss"].append(avg_train_loss.cpu().detach().numpy())
    #     h["test_loss"].append(avg_test_loss.cpu().detach().numpy())
    #     print("[INFO] EPOCH: {}/{}".format(e + 1, 10))
    #     print("Train loss: {:.6f}, Test loss: {:.4f}".format(
    #         avg_train_loss, avg_test_loss))


    #unet = save.load_model("Unet", "48").to(device)


    # pred = predict(test_set[100][0], unet)
    # set = test_set[100]
    # # show_tensor_img(set[0], set[1])
    # s1 = pred.squeeze(0).squeeze(0)
    # s2 = set[1].squeeze(0)
    # print(hausdorff_distance(np.asarray(s1), np.asarray(s2)))
    # intersection = (s1 * s2).sum().float()
    # print((2 * intersection.item()) / (s1.sum() + s2.sum()).item())
    # s1 = np.array(s1)
    # s2 = np.array(s2)
    # dice = np.sum(s1[s2 == 1]) * 2.0 / (np.sum(s1) + np.sum(s2))
    # print(dice)
    # compare_masks(set[0], pred.squeeze(0).squeeze(0), set[1])
    # print(calculate_dice_metric(unet, test_loader, device))


    #print(calculate_mean_distance(unet, test_loader, device))








