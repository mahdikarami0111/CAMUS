import torch


def predict(img, model):
    with torch.no_grad():
        model.eval()
        pred = torch.sigmoid(model(img.unsqueeze(0)))
        return (pred > 0.5).int()
