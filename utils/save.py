import numpy as np
import torch
from models.Unet import Unet
from models.TransUnet.TransUnet import VisionTransformer as TransUnet
from config.TransUnet_cfg import get_TransUnet_config


def save_model(state_dict, name):
    torch.save(state_dict, f"models/trained_models/{name}.pth")


def load_model(model_type, name):
    if model_type == "Unet":
        model = Unet(1, 1)
        model.load_state_dict(torch.load(f"models/trained_models/{name}.pth"))
        return model
    elif model_type == "TransUnet":
        cfg = get_TransUnet_config()
        model = TransUnet(cfg, img_size=224, num_classes=1)
        model.load_state_dict(torch.load(f"models/trained_models/{name}.pth"))
        return model


def load_indices(root="data"):
    return {
        "train": np.loadtxt(f"{root}/train_indices.txt", dtype=int),
        "test": np.loadtxt(f"{root}/test_indices.txt", dtype=int),
        "val": np.loadtxt(f"{root}/val_indices.txt", dtype=int)
    }
