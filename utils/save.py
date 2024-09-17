import numpy as np
import torch
import config.TransUnet_cfg
from models.Unet import Unet
from models.TransUnet.TransUnet import VisionTransformer as TransUnet
from models.BCUnet.BCUnet import BCUnet
from models.DUCKnet.Ducknet import DuckNet


def save_model(state_dict, name):
    torch.save(state_dict, f"models/trained_models/{name}.pth")


def load_model(model_type, name):
    if model_type.lower() == "unet":
        model = Unet(1, 1)

    elif model_type.lower() == "transunet":
        cfg = config.TransUnet_cfg.get_TransUnet_config()
        model = TransUnet(cfg, img_size=224, num_classes=1)

    elif model_type.lower() == "bcunet":
        model = BCUnet(n_channels=1, n_classes=2, bilinear=False)

    elif model_type.lower() == "ducknet":
        model = DuckNet(in_channels=1, out_channels=1, depth=5, init_features=32)

    else:
        raise ValueError("Invalid model type.")
    print(name)
    model.load_state_dict(torch.load(f"models/trained_models/{name}.pth"))
    return model

def load_indices(root="data"):
    return {
        "train": np.loadtxt(f"{root}/train_indices.txt", dtype=int),
        "test": np.loadtxt(f"{root}/test_indices.txt", dtype=int),
        "val": np.loadtxt(f"{root}/val_indices.txt", dtype=int)
    }
