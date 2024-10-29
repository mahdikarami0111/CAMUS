from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.data = {
        "root": "data/database_expanded",
        "device": "cuda",
        "type": "N 4CH",
    }
    config.batch_size = 12
    config.num_classes = 1
    config.image_size = 224
    config.max_epoch = 10
    config.lr = 0.005
    config.train_split = 0.7
    config.test_split = 0.3
    config.val_split = 0
    config.transform = "default"
    return config

