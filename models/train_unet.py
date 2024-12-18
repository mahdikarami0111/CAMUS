import torch
import torch.nn as nn
import os
from train import select_transform
from dataset import CAMUSP, Wrapper
from models.Unet import Unet
from config.TransUnet_cfg import get_TransUnet_config
from torch.utils.data import random_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.nn.modules.loss import BCEWithLogitsLoss
from utils.save import save_model
from torch.nn.functional import sigmoid
import torch.nn.functional as F
from tqdm import tqdm
from utils.losses import DiceLoss


def train(cfg, preset_indices=None):
    batch_size = cfg.batch_size
    num_classes = cfg.num_classes
    image_size = cfg.image_size
    epochs = cfg.max_epoch
    base_lr = cfg.lr

    model = Unet(1, 1).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    bce_loss = BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)

    if preset_indices is None:
        pass
    else:
        train_ = CAMUSP(cfg.data, indices=preset_indices["train"])
        test = CAMUSP(cfg.data, indices=preset_indices["test"])
        val = CAMUSP(cfg.data, indices=preset_indices["val"])

    train_ = Wrapper(train_, select_transform(cfg.transform))
    val = Wrapper(val, select_transform('basic'))
    test = Wrapper(test, select_transform('basic'))

    trainloader = DataLoader(train_, batch_size=batch_size, shuffle=True, pin_memory=True)
    valloader = DataLoader(val, batch_size=batch_size, shuffle=True, pin_memory=True)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=True, pin_memory=True)

    min_loss = 999999
    best_model = None

    for e in tqdm(range(epochs)):
        model.train()
        for i, sampled_batch in enumerate(trainloader):
            image_batch, mask_batch = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
            outputs = model(image_batch)
            loss_ce = bce_loss(outputs, mask_batch)
            loss_dice = dice_loss(outputs, mask_batch)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 25 == 0 or i == len(valloader) - 1:
                print(f"iteration {i}/{len(valloader) - 1} total loss: {loss} | dice: {loss_dice} | bce: {loss_ce}")

        total_loss = 0
        with torch.no_grad():
            model.eval()
            for i, sampled_batch in enumerate(valloader):
                image_batch, mask_batch = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
                outputs = model(image_batch)
                loss_ce = bce_loss(outputs, mask_batch)
                loss_dice = dice_loss(outputs, mask_batch, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                total_loss += loss
            total_loss /= len(val)
            print(f"epoch {e}/{epochs} total loss: {total_loss}")
            if total_loss < min_loss:
                min_loss = total_loss
                save_model(model.state_dict(), f"{e}")
                if best_model is not None:
                    os.remove("models/trained_models/" + best_model+".pth")
                best_model = f"{e}"

