from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.device = 'cuda'
    config.model = 'Unet'
    config.loss = "BCEL"
    config.optimizer = {
        "name": "ADAM",
        "lr": 0.05
    }
    config.scheduler = {
        "name": "EXPONENTIAL",
        "gamma": 0.95
    }
    config.data = {
        "root": "data/database_expanded",
        "device": "cuda",
        "type": "0_4ch"
    }
    config.transform = "default"
    config.train = ConfigDict()
    config.train.train = 0.7
    config.train.test = 0.15
    config.train.val = 0.15
    config.train.batch_size = 8
    config.train.epochs = 10
    config.train.kfold = 10

    return config

