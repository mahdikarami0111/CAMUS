import torch
from models.Unet import Unet


def save_model(state_dict, name):
    torch.save(state_dict, f"models/trained_models/{name}.pth")


def load_model(model_type, name):
    if model_type == "Unet":
        model = Unet(1, 1)
        model.load_state_dict(torch.load(f"models/trained_models/{name}.pth"))
        return model
