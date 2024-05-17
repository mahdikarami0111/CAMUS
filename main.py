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
from utils.debug import*

# img = nib.load("data/database/patient0001/patient0001_4CH_ES.nii")
# nii_data = np.array(np.transpose(img.get_fdata()), dtype=np.uint8)
#
# cv2.imshow("img", nii_data)
# cv2.waitKey(0)
# do_stuff("data/database/patient0001/patient0001_2CH_ES.nii", "data/database/patient0001/patient0001_2CH_ES_gt.nii")
if __name__ == '__main__':
    with open("config/dataset_config.yaml", 'r') as cfg:
        temp = yaml.safe_load(cfg)

    # crop_ratio = 1.0
    input_size = 224
    device = "cuda"
    input_transformer = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=1.0),
        A.Resize(height=input_size, width=input_size),
        ToTensorV2(),
        ])

    dataset = CAMUS(temp, input_transformer)
    train_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.3])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=8, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=8, pin_memory=True)

    unet = Unet(1, 1).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(unet.parameters(), 0.02)
    scheduler = ExponentialLR(opt, gamma=0.95)

    train_steps = len(train_set) // 32
    test_steps = len(test_set) // 32

    h = {"train_loss": [], "test_loss": []}

    for e in tqdm(range(30)):
        unet.train()

        total_train_loss = 0
        total_test_loss = 0

        for(i, (X, Y)) in enumerate(train_loader):
            (X, Y) = (X.to(device), Y.to(device))
            out = unet(X)
            loss = loss_function(out, Y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_train_loss += loss

        with torch.no_grad():
            unet.eval()

            for(x, y) in test_loader:
                (x, y) = (x.to(device), y.to(device))
                out = unet(x)
                total_test_loss += loss_function(out, y)

        avg_train_loss = total_train_loss / train_steps
        avg_test_loss = total_test_loss / test_steps
        scheduler.step()

        h["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        h["test_loss"].append(avg_test_loss.cpu().detach().numpy())
        print("[INFO] EPOCH: {}/{}".format(e + 1, 10))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avg_train_loss, avg_test_loss))
    # unet = save.load_model("Unet", "Unet")
    # pred = predict(test_set[100][0], unet)
    # # show_tensor_img(test_set[100][0], pred.squeeze(0).squeeze(0))
    # compare_masks(test_set[100][0], pred.squeeze(0).squeeze(0), test_set[100][1])
    # save.save_model(unet.state_dict(), "Unet")








