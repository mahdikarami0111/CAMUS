import os

import models.Unet
import torch
from torch.optim.lr_scheduler import ExponentialLR
from dataset import CAMUS
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from sklearn.model_selection import KFold
from utils.save import save_model
from utils.save import load_model
from eval.evaluation import evaluate_model


def select_model(name):
    if name == "Unet":
        return models.Unet.Unet(1, 1)


def select_loss(loss):
    if loss == "BCEL":
        return torch.nn.BCEWithLogitsLoss()


def select_opt(opt, model):
    if opt.NAME == "ADAM":
        return torch.optim.Adam(model.parameters(), opt.LR)


def select_scheduler(scheduler, optimizer):
    if scheduler.NAME == "EXPONENTIAL":
        return ExponentialLR(optimizer, scheduler.GAMMA)


def select_transform(transform):
    if transform == "default":
        input_size = 224
        t = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=1.0),
            A.Resize(height=input_size, width=input_size),
            ToTensorV2(),
        ])
        return t


def train(cfg):
    device = cfg.DEVICE
    model = select_model(cfg.MODEL).to(device)
    loss_function = select_loss(cfg.LOSS)
    opt = select_opt(cfg.OPTIMIZER, model)
    scheduler = select_scheduler(cfg.SCHEDULER, opt)
    dataset = CAMUS(cfg.DATA, select_transform(cfg.TRANSFORM))
    train_set, test_set, val_set = random_split(dataset, [cfg.TRAIN.TRAIN, cfg.TRAIN.VAL, cfg.TRAIN.TEST])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=cfg.TRAIN.BATCH_SIZE, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=cfg.TRAIN.BATCH_SIZE, pin_memory=True)


    train_loader = DataLoader(train_set, shuffle=True, batch_size=cfg.TRAIN.BATCH_SIZE, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=cfg.TRAIN.BATCH_SIZE, pin_memory=True)

    epochs = cfg.TRAIN.EPOCHS

    train_steps = len(train_set) // 8
    val_steps = len(val_set) // 8

    h = {"train_loss": [], "test_loss": []}

    for e in tqdm(range(epochs)):
        model.train()

        total_train_loss = 0
        total_val_loss = 0

        for(i, (X, Y)) in enumerate(train_loader):
            (X, Y) = (X.to(device), Y.to(device))
            out = model(X)
            loss = loss_function(out, Y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_train_loss += loss

        with torch.no_grad():
            model.eval()

            for(x, y) in val_loader:
                (x, y) = (x.to(device), y.to(device))
                out = model(x)
                total_val_loss += loss_function(out, y)

        avg_train_loss = total_train_loss / train_steps
        avg_test_loss = total_val_loss / val_steps
        scheduler.step()

        h["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        h["test_loss"].append(avg_test_loss.cpu().detach().numpy())
        print("[INFO] EPOCH: {}/{}".format(e + 1, 10))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avg_train_loss, avg_test_loss))
    return train_set.indices, val_set.indices, test_set.indices


def train_K_fold(cfg):
    device = cfg.DEVICE
    dataset = CAMUS(cfg.DATA, select_transform(cfg.TRANSFORM))
    kfold = cfg.TRAIN.KFOLD
    epochs = cfg.TRAIN.EPOCHS
    kf = KFold(kfold, shuffle=True)

    for j, (train_index, test_index) in enumerate(kf.split(dataset)):
        print(f"Fold {j}/{kfold}")

        model = select_model(cfg.MODEL).to(device)
        loss_function = select_loss(cfg.LOSS)
        opt = select_opt(cfg.OPTIMIZER, model)
        scheduler = select_scheduler(cfg.SCHEDULER, opt)

        train_val_set = Subset(dataset, train_index)
        test_set = Subset(dataset, test_index)
        train_set, val_set = random_split(train_val_set, [cfg.TRAIN.TRAIN, cfg.TRAIN.VAL])

        train_loader = DataLoader(train_set, shuffle=True, batch_size=cfg.TRAIN.BATCH_SIZE, pin_memory=True)
        val_loader = DataLoader(val_set, shuffle=True, batch_size=cfg.TRAIN.BATCH_SIZE, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=True, batch_size=cfg.TRAIN.BATCH_SIZE, pin_memory=True)

        train_steps = len(train_set) // 8
        val_steps = len(val_set) // 8

        h = {"train_loss": [], "test_loss": []}
        min_loss = 99999
        best_model = None

        for e in tqdm(range(epochs)):
            model.train()

            total_train_loss = 0
            total_val_loss = 0

            for (i, (X, Y)) in enumerate(train_loader):
                (X, Y) = (X.to(device), Y.to(device))
                out = model(X)
                loss = loss_function(out, Y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_train_loss += loss

            with torch.no_grad():
                model.eval()

                for (x, y) in val_loader:
                    (x, y) = (x.to(device), y.to(device))
                    out = model(x)
                    total_val_loss += loss_function(out, y)

            avg_train_loss = total_train_loss / train_steps
            avg_val_loss = total_val_loss / val_steps
            scheduler.step()

            h["train_loss"].append(avg_train_loss.cpu().detach().numpy())
            h["test_loss"].append(avg_val_loss.cpu().detach().numpy())
            if avg_val_loss < min_loss:
                min_loss = avg_val_loss
                save_model(model.state_dict(), f"{j}{e}")
                if best_model is not None:
                    os.remove("models/trained_models/" + best_model+".pth")
                best_model = f"{j}{e}"

            print("[INFO] EPOCH: {}/{}".format(e + 1, 10))
            print("Train loss: {:.6f}, Test loss: {:.4f}".format(
                avg_train_loss, avg_val_loss))
        model = load_model(cfg.MODEL, best_model)
        results = evaluate_model(model, test_loader, device)
        print(results)





